# hipsplit — an adaptive work splitter for HIP

`hipsplit` answers one question, well: *given the GPU(s) actually present — on
this machine, and across the cluster — and a kernel I'm about to launch, how
should I size the grid and split the work so the hardware stays busy?*

You hand it a short description of your kernel (block size, shared-memory use)
and the amount of work, and it hands back a concrete launch plan per device:
block size, grid size, which slice of the work each GPU takes, and an honest
signal of whether the GPU will actually be saturated. If you also want it to
*run* that plan, there's an opt-in executor that does the launches for you.

It's a small, self-contained C++17 module with no ties to any particular
application. Drop it into a ROCm project with `add_subdirectory()`, build and
install it on its own, or load the shared library over FFI from another
language. It leans only on stock HIP APIs (`hipGetDeviceProperties`,
`hipOccupancyMaxActiveBlocksPerMultiprocessor`,
`hipOccupancyMaxPotentialBlockSize`, plain stream calls), nothing exotic, so
it's a clean candidate for upstreaming into the ROCm ecosystem.

---

## Why you'd want this

Picking a launch configuration by hand is fiddly and machine-specific. A fixed
`grid = 80, block = 256` that you tuned on one card is wrong on the next one:
different compute-unit counts, different shared-memory budgets, different
wavefront sizes. And the intuitive "one chunk per wavefront" mental model is
backwards — GPUs hide memory latency by *oversubscribing* each compute unit with
many more wavefronts than it can run at once, then switching between them.

`hipsplit` does the arithmetic for you, from the real device properties, every
time the program runs. Same code, sensible launch on any AMD GPU with ROCm —
one card, several cards in a box, or many nodes in a rack.

---

## How it's structured

The module is split in two on purpose:

- **`hipsplit::core`** — the planner. Pure C++, **no HIP dependency**. It takes a
  value-type description of the devices (`Topology`) plus the kernel and
  workload, and produces a `LaunchPlan`. Because it never touches a GPU, it
  builds and unit-tests on any machine, including CI runners with no Radeon card.
- **`hipsplit::hip`** — a thin HIP layer. It fills a `Topology` from the live
  runtime (`query_topology`), wraps HIP's occupancy calculator, offers the
  one-call `plan_auto()`, and — if you ask for it — the executor.

All the logic worth testing lives in the part that doesn't need hardware. That's
what keeps it trustworthy.

---

## Out of the box

The shortest path — let it find the GPUs and plan in one call:

```cpp
#include <hipsplit/hipsplit.hpp>

hipsplit::KernelProfile kp;
kp.threads_per_block = 256;        // or leave 0 to let hipsplit choose
kp.dynamic_lds_bytes = 32 * 1024;  // shared memory your kernel asks for, per block

hipsplit::Workload w;
w.total_items = batch * heads * banks * tokens;  // independent units of work

hipsplit::LaunchPlan plan = hipsplit::plan_auto(kp, w);
if (!plan.valid) {
    // no GPU visible (or no work for this node) — plan.note says why
}

for (const auto& d : plan.launches) {
    hipSetDevice(d.device_id);
    my_kernel<<<d.grid_size, d.block_size, kp.dynamic_lds_bytes>>>(
        /* base   = */ d.item_offset,
        /* count  = */ d.item_count,
        /* ...your buffers... */);
}
```

That's the whole integration: describe the kernel, describe the work, launch
what the plan tells you over the slice it assigns. (Or hand the plan to the
executor and skip the loop — see below.)

If you'd rather inspect or fake the topology (testing, a fixed device subset,
what-if sizing), build it yourself and call the planner directly:

```cpp
hipsplit::Topology topo = hipsplit::query_topology();  // or hand-build it
hipsplit::LaunchPlan plan = hipsplit::Planner::plan(topo, kp, w);
```

---

## Multi-GPU and multi-node

The split is **built into the planner** — there's no separate cluster API. The
same `plan()` / `plan_auto()` handle one card, several cards, or many nodes;
you just describe more in the `Topology`:

- **One GPU** — the plan has a single launch covering all the work.
- **Several GPUs in one box** — the work is divided across them in proportion to
  each device's `compute_weight()` (CU count, nudged by clock), rounded to
  `split_granularity`. Each slice is contiguous; nothing is lost or duplicated.
- **Many nodes** — set `topo.num_nodes` and `topo.this_node_rank`. The planner
  first divides the work across nodes (equally, or by `topo.node_weights` if you
  provide them), then divides *this node's* slice across its local GPUs. The
  node-level split is deterministic and identical on every rank, so each node
  independently agrees on who owns which contiguous global range. The plan
  reports this node's range in `node_item_offset` / `node_item_count`, and every
  launch carries a global offset inside it.

`plan_auto()` will pick up a cluster from the environment so the one-call path
works under a multi-node launcher with no code change:

```
HIPSPLIT_NUM_NODES=4   HIPSPLIT_NODE_RANK=2   ./your_program
```

With those unset it plans as a single machine. The names are deliberately our
own — a framework's global *process* rank is not the same as a *node* rank, and
we won't guess.

**Honest scope:** the planner only decides the global tiling — it tells each
node (and each GPU) which items are theirs. It does **not** move data between
nodes. Cross-node exchange of activations/gradients is your transport's job
(RCCL/MPI), because it's inseparable from your tensors. hipsplit won't pretend
otherwise.

### Discovering a cluster over MPI (opt-in)

Filling in `num_nodes` / `this_node_rank` / `node_weights` by hand is fine, but
if you already run under MPI there's a helper that does it for you. It lives
behind `-DHIPSPLIT_WITH_MPI=ON`, in its own target (`hipsplit::mpi`) and its own
header, so neither MPI nor the dependency ever leaks into the core or the
default build:

```cpp
#include <hipsplit/cluster_mpi.hpp>

hipsplit::Topology topo = hipsplit::query_cluster_topology(MPI_COMM_WORLD);
hipsplit::LaunchPlan plan = hipsplit::Planner::plan(topo, kp, w);
```

Each rank contributes the summed compute weight of its local GPUs; the helper
gathers those across the communicator and fills the cluster fields, so the
planner divides work between nodes in proportion to their real muscle.

It assumes **one MPI rank per node** — the natural shape for this node → device
split. A rank that sees no GPU contributes zero weight and is simply handed no
work. And, as everywhere in hipsplit, the helper only *discovers topology and
decides the tiling*; it moves no data. The cross-node exchange stays with your
transport.

---

## Actually running it: the executor (opt-in)

The planner is advice-only. If you want that advice executed, include the
executor — it's a **separate header** so you only take the launch dependency
when you ask for it:

```cpp
#include <hipsplit/executor.hpp>

hipsplit::StreamExecutor exec;
hipsplit::RunResult r = exec.submit(plan,
    [&](const hipsplit::DeviceLaunch& s, hipStream_t stream) {
        my_kernel<<<s.grid_size, s.block_size, kp.dynamic_lds_bytes, stream>>>(
            data + s.item_offset, s.item_count);
    });
if (!r.ok) { /* inspect r.devices for the offending device_id + hipError_t */ }
```

The argument-passing problem — the genuinely tricky part of any launcher — is
solved with a **callable**. Kernel signatures are arbitrary and typed; a generic
`void**`-args interface would erase those types and invite mistakes. Instead
your lambda captures the real, typed kernel and buffers, and the executor stays
completely kernel-agnostic. You keep full control of the launch.

What the executor owns, and nothing more: for each slice it selects the device,
hands you a **per-device stream** (so the GPUs overlap rather than running one
after another), invokes your lambda, and after everything is in flight it does a
single synchronize across all the streams. It reports per-device status.

It's a **single-node, multi-GPU** executor — it drives the devices this process
can see. On a multi-node plan it runs this node's slice; coordinating the other
nodes (and moving data between them) is still your transport's job.

FFI callers don't get the executor (a C++ lambda can't cross the C boundary):
from another language, take the plan over the C ABI and run the short launch
loop yourself in your own language.

---

## API reference

### `DeviceInfo` — one GPU, as plain data

A HIP-free value type. Filled in by `query_topology()` from `hipDeviceProp_t`, or
hand-built in a test. Key fields:

| field | meaning | source |
|---|---|---|
| `device_id` | HIP ordinal to pass to `hipSetDevice` | — |
| `name`, `arch` | e.g. `"AMD Radeon Graphics"`, `"gfx1151"` | `name`, `gcnArchName` |
| `compute_units` | CU count | `multiProcessorCount` |
| `max_threads_per_block` | hardware block ceiling | `maxThreadsPerBlock` |
| `max_threads_per_cu` | resident-thread ceiling per CU | `maxThreadsPerMultiProcessor` |
| `wavefront_size` | 32 or 64 on RDNA | `warpSize` |
| `lds_per_block`, `lds_per_cu` | shared-memory budgets, in bytes | `sharedMemPerBlock` |
| `global_mem` | device memory, in bytes | `totalGlobalMem` |
| `clock_khz` | core clock, used only as a split tie-breaker | `clockRate` |
| `max_blocks_per_cu` | upper clamp on resident blocks per CU | default `32` |

`compute_weight()` returns the relative throughput weight used when splitting
work across several GPUs (and, via the MPI helper, across nodes).

### `Topology` — the GPUs you're allowed to use

| field | meaning |
|---|---|
| `devices` | this node's local GPUs |
| `this_node_rank` | which node we are (0-based); default 0 |
| `num_nodes` | nodes in the cluster; default 1 (disables node-level split) |
| `node_weights` | optional per-node relative weight, length `num_nodes`; empty = equal |

Build it with `query_topology()` (fills `devices`), `query_cluster_topology()`
(fills the cluster fields too, over MPI), or fill it yourself.

### `KernelProfile` — your kernel, described

| field | meaning |
|---|---|
| `threads_per_block` | fixed block size, or `0` to let the planner pick (~256, clamped) |
| `dynamic_lds_bytes` | dynamic shared memory per block |
| `static_lds_bytes` | static shared memory per block |
| `block_size_limit` | upper bound when auto-picking |
| `block_size_multiple` | round the auto-picked size to this (defaults to the wavefront size) |

### `Workload` — the work, described

| field | meaning |
|---|---|
| `total_items` | number of independent units of work |
| `items_per_thread` | how many units one thread handles (default 1) |
| `split_granularity` | only cut between GPUs/nodes on this boundary (e.g. a whole head) |

### `DeviceLaunch` / `LaunchPlan` — what to launch

`LaunchPlan` holds `valid`, a human-readable `note`, this node's
`node_item_offset` / `node_item_count`, and a `launches` vector of
`DeviceLaunch`, one per local device that got work:

| field | meaning |
|---|---|
| `device_id` | which GPU |
| `item_offset`, `item_count` | the half-open slice `[offset, offset + count)` this GPU owns |
| `block_size`, `grid_size` | launch dimensions |
| `blocks_per_cu` | estimated resident blocks per CU |
| `saturation` | `grid_size` vs. what the device can keep busy (see below) |

### Functions

- `LaunchPlan plan_auto(const KernelProfile&, const Workload&)` — query the live
  devices (and any cluster from the environment) and plan, in one call. *(HIP layer.)*
- `Topology query_topology()` / `DeviceInfo query_device(int)` — read the live
  HIP devices. *(HIP layer.)*
- `Topology query_cluster_topology(MPI_Comm = MPI_COMM_WORLD)` — discover the
  cluster over MPI and fill the node fields. *(Opt-in MPI helper, `hipsplit::mpi`.)*
- `LaunchPlan Planner::plan(const Topology&, const KernelProfile&, const Workload&)`
  — the pure planner, single- or multi-node. *(Core, no HIP.)*
- `Planner::choose_block_size`, `Planner::estimate_blocks_per_cu`,
  `Planner::plan_one` — the building blocks, exposed for testing and fine control.
- `int occupancy_blocks_per_cu(const void* kernel, int block_size, size_t dyn_lds)`
  and `bool suggest_block_size(...)` — thin wrappers over HIP's occupancy
  calculator, for when you have a real kernel pointer and want the runtime's own
  numbers (these also account for register pressure). *(HIP layer.)*
- `StreamExecutor::submit(const LaunchPlan&, LaunchFn)` — run a plan across the
  local GPUs, one stream per device, and wait. *(HIP layer, opt-in header.)*
- C ABI: `hipsplit_plan_auto(...)` plus `hipsplit_kernel_profile_default()` /
  `hipsplit_workload_default()`, declared in `hipsplit/hipsplit_c.h`, for FFI.

---

## How the planner thinks

**Block size.** If you set `threads_per_block`, it's rounded to a wavefront
multiple and clamped to the hardware limit. If you leave it `0`, the planner aims
for 256 threads — a good balance on RDNA between latency hiding and per-block
resources — then clamps to your limits. With a real kernel pointer, prefer
`suggest_block_size()` for the runtime's occupancy-tuned answer.

**Occupancy.** `estimate_blocks_per_cu()` works out how many blocks fit on one
CU, limited by the thread budget (`max_threads_per_cu / block_size`) and the
shared-memory budget (`lds_per_cu / lds_per_block`). A block that asks for more
LDS than a workgroup is allowed returns a hard zero — a real, useful failure
signal rather than a silent bad launch.

**Splitting.** Work is divided — across nodes first (if any), then across each
node's GPUs — in proportion to weight, rounded to `split_granularity`, with any
remainder handed to the first bucket so nothing is ever lost or duplicated.
Every slice is contiguous.

**Saturation.** `saturation = grid_size / (blocks_per_cu * compute_units)`. At
`>= 1.0` you have enough blocks to keep the device busy and hide latency. Below
`1.0` the GPU is starved — the workload is too small to fill it — and the plan's
`note` says so. It's a diagnostic, not an error.

---

## Build

```
cmake -S . -B build
cmake --build build
ctest --test-dir build              # planner tests (any machine) + executor smoke test (skips with no GPU)
./build/hipsplit_print_topology     # GPU smoke check (needs ROCm + a card)
```

The planner core builds with any C++17 compiler. The HIP layer builds when CMake
can find ROCm: it looks under `-DROCM_PATH`, then `$ROCM_PATH`, then `/opt/rocm`,
and falls back to locating `hip/hip_runtime.h` + `libamdhip64` directly if the
full HIP CMake package isn't installed. On a box with no ROCm at all, only the
core (and its tests) are built.

Useful options:

- `-DHIPSPLIT_WITH_HIP=OFF` — build only the planner core.
- `-DHIPSPLIT_BUILD_SHARED=OFF` — skip the shared FFI library.
- `-DHIPSPLIT_WITH_MPI=ON` — build the optional MPI cluster-discovery helper (needs MPI).
- `-DHIPSPLIT_BUILD_TESTS=OFF` — skip the tests.
- `-DROCM_PATH=/path/to/rocm` — point at a non-standard ROCm install.

### Sample `print_topology` output (gfx1151)

```
found 1 HIP device(s):
  [0] AMD Radeon Graphics (gfx1151)
      CUs=20  threads/block=1024  threads/CU=2048  wavefront=32
      LDS/block=65536 B  global=65536 MiB  clock=2900000 kHz

plan for 524288 items (ok):
  dev 0: items [0, 524288)  grid=2048  block=256  blocks/CU=2  sat=51.20
```

Here a 32 KB-LDS kernel is capped at 2 resident blocks per CU by shared memory,
and the 2048-block grid oversubscribes the 20 CUs ~51x — plenty to hide latency.

---

## Using it from another build

```cmake
add_subdirectory(external/hip_adaptive_splitter)

# Link the core anywhere (no HIP needed):
target_link_libraries(my_planner_only PRIVATE hipsplit::core)

# Link the HIP layer where you actually launch kernels (planner + executor):
target_link_libraries(my_gpu_app PRIVATE hipsplit::hip)

# Opt into the MPI cluster helper (configure with -DHIPSPLIT_WITH_MPI=ON):
target_link_libraries(my_cluster_app PRIVATE hipsplit::mpi)
```

From a non-C++ codebase, load the shared library (`libhipsplit_c`) and call the
C ABI in `hipsplit/hipsplit_c.h` — a C ABI is callable from Rust, Python
(ctypes/cffi), Go (cgo), Node, Julia, and plain C alike.

---

## Roadmap

- [x] Device topology value types + GPU-free launch planner
- [x] Unit tests for the planner
- [x] HIP device query + occupancy layer
- [x] `plan_auto()` one-call entry point
- [x] Print-topology smoke utility
- [x] C ABI + shared library for FFI consumers
- [x] Multi-node splitting built into the planner
- [x] StreamExecutor for overlapping per-device launches
- [x] GPU-guarded executor smoke test (skips without a card)
- [x] Optional MPI cluster-topology discovery helper, kept out of the core

## License & contributing

Kept intentionally dependency-light and ROCm-idiomatic so it can be proposed
upstream. Contributions and porting reports from other AMD cards are welcome —
the planner tests run without a GPU, so they're easy to extend.
