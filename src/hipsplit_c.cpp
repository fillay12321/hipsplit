#include "hipsplit/hipsplit_c.h"

#include "hipsplit/autoplan.hpp"
#include "hipsplit/planner.hpp"

#include <algorithm>
#include <cstring>
#include <string>

// The C ABI is a thin marshalling layer: copy the plain-C structs into the C++
// value types, run the real planner, then copy the result back into
// caller-owned buffers. No C++ object ever escapes across the boundary.

using hipsplit::DeviceLaunch;
using hipsplit::KernelProfile;
using hipsplit::LaunchPlan;
using hipsplit::Workload;

extern "C" {

hipsplit_kernel_profile hipsplit_kernel_profile_default(void) {
    KernelProfile def;
    hipsplit_kernel_profile out;
    out.threads_per_block   = def.threads_per_block;
    out.dynamic_lds_bytes   = def.dynamic_lds_bytes;
    out.static_lds_bytes    = def.static_lds_bytes;
    out.block_size_limit    = def.block_size_limit;
    out.block_size_multiple = def.block_size_multiple;
    return out;
}

hipsplit_workload hipsplit_workload_default(void) {
    Workload def;
    hipsplit_workload out;
    out.total_items       = def.total_items;
    out.items_per_thread  = def.items_per_thread;
    out.split_granularity = def.split_granularity;
    return out;
}

int hipsplit_plan_auto(const hipsplit_kernel_profile* kp_c,
                       const hipsplit_workload*       w_c,
                       hipsplit_device_launch*        out_launches,
                       int                            launch_cap,
                       int*                           out_valid,
                       char*                          out_note,
                       int                            note_cap) {
    KernelProfile kp;
    if (kp_c) {
        kp.threads_per_block   = kp_c->threads_per_block;
        kp.dynamic_lds_bytes   = kp_c->dynamic_lds_bytes;
        kp.static_lds_bytes    = kp_c->static_lds_bytes;
        kp.block_size_limit    = kp_c->block_size_limit;
        kp.block_size_multiple = kp_c->block_size_multiple;
    }

    Workload w;
    if (w_c) {
        w.total_items       = w_c->total_items;
        w.items_per_thread  = w_c->items_per_thread;
        w.split_granularity = w_c->split_granularity;
    }

    LaunchPlan plan = hipsplit::plan_auto(kp, w);

    if (out_valid) {
        *out_valid = plan.valid ? 1 : 0;
    }

    if (out_note && note_cap > 0) {
        const std::string& note = plan.note;
        const int copy = static_cast<int>(
            std::min<std::size_t>(note.size(),
                                  static_cast<std::size_t>(note_cap - 1)));
        if (copy > 0) {
            std::memcpy(out_note, note.data(), static_cast<std::size_t>(copy));
        }
        out_note[copy] = '\0';
    }

    const int total = static_cast<int>(plan.launches.size());
    if (out_launches && launch_cap > 0) {
        const int copy = std::min(total, launch_cap);
        for (int i = 0; i < copy; ++i) {
            const DeviceLaunch&     src = plan.launches[static_cast<std::size_t>(i)];
            hipsplit_device_launch& dst = out_launches[i];
            dst.device_id     = src.device_id;
            dst.item_offset   = src.item_offset;
            dst.item_count    = src.item_count;
            dst.block_size    = src.block_size;
            dst.grid_size     = src.grid_size;
            dst.blocks_per_cu = src.blocks_per_cu;
            dst.saturation    = src.saturation;
        }
    }

    return total;
}

} // extern "C"
