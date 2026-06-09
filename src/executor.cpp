#include "hipsplit/executor.hpp"

namespace hipsplit {

StreamExecutor::~StreamExecutor() {
    for (auto& s : streams_) {
        if (s.second) {
            // Destroy each stream on the device it belongs to. Errors here are
            // not actionable during teardown, so they're intentionally ignored.
            hipSetDevice(s.first);
            hipStreamDestroy(s.second);
        }
    }
}

hipStream_t StreamExecutor::stream_for(int device_id) {
    for (auto& s : streams_) {
        if (s.first == device_id) return s.second;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        // No stream: fall back to the default stream. We still record it so the
        // synchronize phase waits on the right thing.
        stream = nullptr;
    }
    streams_.push_back({device_id, stream});
    return stream;
}

RunResult StreamExecutor::submit(const LaunchPlan& plan, const LaunchFn& launch) {
    RunResult res;

    if (!plan.valid || plan.launches.empty()) {
        res.ok   = false;
        res.note = !plan.valid
            ? (plan.note.empty() ? "plan is not valid" : plan.note)
            : "plan has no launches";
        return res;
    }
    if (!launch) {
        res.ok   = false;
        res.note = "no launch callback provided";
        return res;
    }

    // Phase 1: fire every slice on its own device and stream so the devices run
    // concurrently instead of one-after-another.
    std::vector<std::pair<int, hipStream_t>> issued;
    issued.reserve(plan.launches.size());
    for (const auto& slice : plan.launches) {
        const hipError_t set_err = hipSetDevice(slice.device_id);
        if (set_err != hipSuccess) {
            res.ok = false;
            res.devices.push_back({slice.device_id, set_err});
            continue;
        }

        hipStream_t stream = stream_for(slice.device_id);

        // Clear any stale error so a launch failure is attributed to this slice.
        hipGetLastError();
        launch(slice, stream);
        const hipError_t launch_err = hipGetLastError();
        if (launch_err != hipSuccess) {
            res.ok = false;
            res.devices.push_back({slice.device_id, launch_err});
        }

        issued.push_back({slice.device_id, stream});
    }

    // Phase 2: now that everything is in flight, wait for each device to drain.
    for (const auto& it : issued) {
        if (hipSetDevice(it.first) != hipSuccess) {
            res.ok = false;
            continue;
        }
        const hipError_t sync_err = hipStreamSynchronize(it.second);
        if (sync_err != hipSuccess) {
            res.ok = false;
            res.devices.push_back({it.first, sync_err});
        }
    }

    return res;
}

} // namespace hipsplit
