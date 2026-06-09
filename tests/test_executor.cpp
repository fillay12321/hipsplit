// GPU-guarded smoke test for the StreamExecutor.
//
// Unlike the planner tests (which are pure and run anywhere), this one needs a
// real HIP device. When none is visible -- e.g. CI without a Radeon card -- it
// prints a skip line and returns cleanly, so the suite stays green. With a card
// present it drives the whole execution path: plan -> per-device stream ->
// user launch -> synchronize.
//
// The "kernel" here is just an asynchronous memset over each slice. That is
// enough to exercise device selection, the per-device stream and the final
// sync, and it compiles with a plain host compiler -- no separately compiled
// kernel (and therefore no hipcc) required.

#include "hipsplit/hipsplit.hpp"
#include "hipsplit/executor.hpp"
#include "microtest.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdio>
#include <vector>

using namespace hipsplit;

TEST(executor_runs_a_plan_on_real_devices) {
    Topology topo = query_topology();
    if (topo.empty()) {
        std::printf("  [skip] no HIP device visible; executor smoke test skipped\n");
        return;
    }

    KernelProfile kp;
    kp.threads_per_block = 256;

    Workload w;
    w.total_items = 1u << 20; // 1Mi independent units of work

    LaunchPlan plan = Planner::plan(topo, kp, w);
    CHECK(plan.valid);
    CHECK(!plan.launches.empty());

    // One device buffer per slice, sized to that slice. Bytes == item_count is
    // arbitrary but fine: we only care that the async op runs on the right
    // device and stream.
    struct Buf {
        int            device_id = 0;
        unsigned char* ptr       = nullptr;
        std::size_t    bytes     = 0;
    };
    std::vector<Buf> bufs;
    bufs.reserve(plan.launches.size());
    for (const auto& l : plan.launches) {
        if (l.item_count == 0) continue;
        Buf b;
        b.device_id = l.device_id;
        b.bytes     = static_cast<std::size_t>(l.item_count);
        CHECK_EQ(hipSetDevice(l.device_id), hipSuccess);
        CHECK_EQ(hipMalloc(&b.ptr, b.bytes), hipSuccess);
        bufs.push_back(b);
    }

    // The launch is a callable: it owns the typed work, the executor stays
    // kernel-agnostic. Match each slice to its buffer by device id.
    StreamExecutor exec;
    RunResult r = exec.submit(plan, [&](const DeviceLaunch& s, hipStream_t stream) {
        for (const auto& b : bufs) {
            if (b.device_id == s.device_id) {
                hipMemsetAsync(b.ptr, 0, b.bytes, stream);
                break;
            }
        }
    });

    CHECK(r.ok);
    CHECK(r.devices.empty()); // no per-device errors recorded on a clean run

    for (auto& b : bufs) {
        hipSetDevice(b.device_id);
        hipFree(b.ptr);
    }
}

int main() {
    return microtest::run_all();
}
