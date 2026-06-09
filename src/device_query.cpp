#include "hipsplit/device_query.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace hipsplit {
namespace {

DeviceInfo from_props(int id, const hipDeviceProp_t& p) {
    DeviceInfo d;
    d.device_id             = id;
    d.name                  = p.name;
    d.arch                  = p.gcnArchName;
    d.compute_units         = p.multiProcessorCount;
    d.max_threads_per_block = p.maxThreadsPerBlock;
    d.max_threads_per_cu    = p.maxThreadsPerMultiProcessor;
    d.wavefront_size        = p.warpSize;
    d.lds_per_block         = p.sharedMemPerBlock;
    // hipDeviceProp_t has no portable "LDS per CU" figure across every ROCm
    // release, and on RDNA a CU shares roughly one workgroup's worth of LDS
    // anyway, so the per-block limit is a safe stand-in. A caller who knows the
    // real per-CU budget can overwrite lds_per_cu before planning.
    d.lds_per_cu            = p.sharedMemPerBlock;
    d.global_mem            = p.totalGlobalMem;
    d.clock_khz             = p.clockRate;
    return d;
}

} // namespace

DeviceInfo query_device(int device_id) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, device_id);
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("hipGetDeviceProperties failed: ") +
                                 hipGetErrorString(err));
    }
    return from_props(device_id, props);
}

Topology query_topology() {
    Topology topo;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) {
        return topo; // no usable HIP devices; caller can fall back to CPU
    }
    topo.devices.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, i) != hipSuccess) continue;
        topo.devices.push_back(from_props(i, props));
    }
    return topo;
}

} // namespace hipsplit
