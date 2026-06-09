#ifndef HIPSPLIT_EXECUTOR_HPP
#define HIPSPLIT_EXECUTOR_HPP

// Opt-in execution layer. The planner only ever tells you *how* to launch; this
// turns that plan into actual launches across the local GPUs. It is kept out of
// the umbrella header on purpose: pull it in explicitly when you want it, and
// you only then take on the HIP-launch dependency.
//
//   hipsplit::StreamExecutor exec;
//   auto status = exec.submit(plan, [&](const hipsplit::DeviceLaunch& s,
//                                       hipStream_t stream) {
//       my_kernel<<<s.grid_size, s.block_size, dyn_lds, stream>>>(
//           data + s.item_offset, s.item_count);
//   });
//
// You write the launch; the executor handles device selection, one stream per
// device so the GPUs overlap, and waiting for all of them. This is a
// single-node, multi-GPU executor: it drives the devices this process can see.
// A multi-node plan still reports this node's slice (LaunchPlan::node_item_*),
// but moving data between nodes is your transport's job (RCCL/MPI), not this
// layer's.

#include "hipsplit/planner.hpp"

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

namespace hipsplit {

// How one device fared. error is the first non-success HIP status seen for it,
// whether from selecting the device, the launch, or the synchronize.
struct DeviceRunStatus {
    int        device_id = 0;
    hipError_t error     = hipSuccess;
};

struct RunResult {
    bool                         ok = true; // true only if every device was clean
    std::vector<DeviceRunStatus> devices;   // one entry per device that erred
    std::string                  note;
};

// Owns a small pool of per-device streams and runs launch plans on them. Reused
// across submit() calls: streams are created lazily per device and torn down
// when the executor is destroyed.
class StreamExecutor {
public:
    StreamExecutor() = default;
    ~StreamExecutor();

    StreamExecutor(const StreamExecutor&)            = delete;
    StreamExecutor& operator=(const StreamExecutor&) = delete;

    // Invoked once per slice, with the slice's device already current and a
    // stream to launch onto. Issue your kernel asynchronously on that stream.
    using LaunchFn = std::function<void(const DeviceLaunch& slice, hipStream_t stream)>;

    // Issue every slice in the plan — one stream per device so devices overlap
    // — then wait for all of them. Returns per-device status; ok is true only if
    // nothing went wrong anywhere.
    RunResult submit(const LaunchPlan& plan, const LaunchFn& launch);

private:
    // (device_id, stream) pairs, one stream per device, created on first use.
    std::vector<std::pair<int, hipStream_t>> streams_;

    // Return this device's stream, creating it if needed. Assumes the device is
    // already current. Falls back to the default stream (nullptr) if creation
    // fails, which costs overlap but still runs correctly.
    hipStream_t stream_for(int device_id);
};

} // namespace hipsplit

#endif // HIPSPLIT_EXECUTOR_HPP
