#ifndef HIPSPLIT_PLANNER_HPP
#define HIPSPLIT_PLANNER_HPP

#include "hipsplit/topology.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hipsplit {

// Describes the kernel we're about to launch, in terms the planner can reason
// about without ever seeing the kernel itself. Leave threads_per_block at 0 to
// let the planner pick a block size.
struct KernelProfile {
    int         threads_per_block   = 0;    // 0 => auto
    std::size_t dynamic_lds_bytes   = 0;    // per-block dynamic shared memory
    std::size_t static_lds_bytes    = 0;    // per-block static shared memory
    int         block_size_limit    = 1024; // upper bound when auto-picking
    // When auto-picking, the chosen block size is always a multiple of this.
    // Leave at 0 to fall back to the device wavefront size.
    int         block_size_multiple = 0;
};

// The work to spread out. total_items is the number of independent units of
// work (for an attention backward pass that might be batch * heads * banks *
// tokens). items_per_thread lets one thread chew through more than one unit,
// which is often the cheapest way to cut launch overhead.
struct Workload {
    std::uint64_t total_items       = 0;
    int           items_per_thread  = 1;
    // Work is only ever cut on this boundary when splitting across GPUs. Useful
    // when consecutive items must stay together (e.g. a whole head per slice).
    std::uint64_t split_granularity = 1;
};

// What to actually launch on one device for its slice of the work.
struct DeviceLaunch {
    int           device_id     = 0;
    std::uint64_t item_offset   = 0;  // first global item handled here
    std::uint64_t item_count    = 0;  // number of items handled here
    int           block_size    = 0;  // threads per block
    std::uint64_t grid_size     = 0;  // number of blocks
    int           blocks_per_cu = 0;  // estimated resident blocks per CU
    // grid_size relative to what the device could keep busy. >= 1.0 means we
    // have enough blocks to hide latency; < 1.0 means the GPU is starved.
    double        saturation    = 0.0;
};

struct LaunchPlan {
    std::vector<DeviceLaunch> launches;
    bool        valid = false;
    std::string note;  // human-readable reason when something had to be clamped

    // This node's slice of the global workload. For a single-node plan this is
    // simply [0, total_items). For a multi-node plan it's the contiguous global
    // range this node is responsible for; every DeviceLaunch above carries a
    // global offset that falls inside this range. Handy for setting up the
    // node's data transfers before launching.
    std::uint64_t node_item_offset = 0;
    std::uint64_t node_item_count  = 0;
};

// Pure, deterministic, GPU-free. The same inputs always produce the same plan,
// which is exactly what makes it cheap to unit test.
class Planner {
public:
    // Pick a block size for one device given the kernel's shared-memory appetite.
    // Returns a multiple of the wavefront size, clamped to hardware and profile
    // limits.
    static int choose_block_size(const DeviceInfo& dev, const KernelProfile& kp);

    // Estimate how many of these blocks fit on one CU at once. This is the
    // analytical fallback for when we can't ask HIP's occupancy API directly.
    static int estimate_blocks_per_cu(const DeviceInfo& dev,
                                      int block_size,
                                      std::size_t total_lds_per_block);

    // Plan a launch for one device handling [offset, offset + count) of the work.
    static DeviceLaunch plan_one(const DeviceInfo& dev,
                                 const KernelProfile& kp,
                                 const Workload& w,
                                 std::uint64_t item_offset,
                                 std::uint64_t item_count);

    // Split the whole workload across the cluster and plan this node's share.
    // When topo.num_nodes > 1 the work is first divided across nodes (by
    // topo.node_weights, or equally), then this node's slice is divided across
    // its local devices weighted by each device's compute_weight(). With a
    // single-node topology this is just the device-level split.
    static LaunchPlan plan(const Topology& topo,
                           const KernelProfile& kp,
                           const Workload& w);
};

} // namespace hipsplit

#endif // HIPSPLIT_PLANNER_HPP
