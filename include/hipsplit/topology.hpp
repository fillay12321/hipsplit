#ifndef HIPSPLIT_TOPOLOGY_HPP
#define HIPSPLIT_TOPOLOGY_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hipsplit {

// Everything the planner needs to know about one GPU. This is a plain value
// type on purpose: it carries no HIP handles, so it can be filled in from a
// real device query or hand-built in a test. Keeping it HIP-free is what lets
// the planning logic compile and run on a machine with no GPU at all.
struct DeviceInfo {
    int         device_id             = 0;
    std::string name;                       // marketing name, e.g. "AMD Radeon 8060S"
    std::string arch;                       // LLVM target, e.g. "gfx1151"

    int         compute_units         = 0;    // CUs (hipDeviceProp_t::multiProcessorCount)
    int         max_threads_per_block = 1024;
    int         max_threads_per_cu    = 0;    // maxThreadsPerMultiProcessor
    int         wavefront_size        = 64;   // 32 or 64 on RDNA (warpSize)

    std::size_t lds_per_block         = 65536; // sharedMemPerBlock (per workgroup)
    std::size_t lds_per_cu            = 65536; // LDS a CU can hand out at once
    std::size_t global_mem            = 0;     // totalGlobalMem

    int         clock_khz             = 0;     // core clock, used only as a weight tie-breaker

    // Rough hardware ceiling on resident workgroups per CU. The scheduler can
    // only keep so many blocks live at once; we clamp derived occupancy with
    // this so the planner never promises more than the hardware can hold.
    int         max_blocks_per_cu     = 32;

    // Relative throughput weight used when splitting work across several GPUs.
    // CU count is a good proxy for homogeneous racks and a sane first guess for
    // mixed ones; clock nudges the tie-break when two parts differ only there.
    double compute_weight() const {
        double w = static_cast<double>(compute_units > 0 ? compute_units : 1);
        if (clock_khz > 0) w *= static_cast<double>(clock_khz) / 1.0e6; // ~GHz
        return w;
    }
};

// The set of GPUs we are allowed to use. The vector index is just our own
// ordering; the real HIP ordinal lives in DeviceInfo::device_id.
//
// Multi-node lives here, not in a separate API. `devices` is always this node's
// LOCAL GPUs. The extra fields describe where this node sits in a cluster; left
// at their defaults they describe a plain single-node machine and the planner
// behaves exactly as it would without them.
struct Topology {
    std::vector<DeviceInfo> devices;

    // Which node we are (0-based) and how many nodes the cluster has. num_nodes
    // == 1 means "just this machine" and disables all node-level splitting.
    int this_node_rank = 0;
    int num_nodes      = 1;

    // Optional per-node relative weight, length == num_nodes. Empty means every
    // node is treated as equal — the right default for a homogeneous rack. Fill
    // it in when nodes differ (e.g. different GPU counts) to bias the split. The
    // node-level split is computed identically on every rank, so all nodes agree
    // on who owns which contiguous slice of the global work.
    std::vector<double> node_weights;

    bool        empty() const { return devices.empty(); }
    std::size_t size()  const { return devices.size(); }
};

} // namespace hipsplit

#endif // HIPSPLIT_TOPOLOGY_HPP
