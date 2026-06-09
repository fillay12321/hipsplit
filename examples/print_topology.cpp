// Smoke utility: enumerate the HIP devices this machine exposes and print a
// sample split plan. It's the quickest way to confirm the HIP layer links
// against ROCm and that the detected numbers look right on a new card.

#include "hipsplit/hipsplit.hpp"

#include <cstdio>

int main() {
    hipsplit::Topology topo = hipsplit::query_topology();
    if (topo.empty()) {
        std::printf("no HIP devices found\n");
        return 0;
    }

    std::printf("found %zu HIP device(s):\n", topo.size());
    for (const auto& d : topo.devices) {
        std::printf("  [%d] %s (%s)\n", d.device_id, d.name.c_str(), d.arch.c_str());
        std::printf("      CUs=%d  threads/block=%d  threads/CU=%d  wavefront=%d\n",
                    d.compute_units, d.max_threads_per_block,
                    d.max_threads_per_cu, d.wavefront_size);
        std::printf("      LDS/block=%zu B  global=%zu MiB  clock=%d kHz\n",
                    d.lds_per_block, d.global_mem / (1024 * 1024), d.clock_khz);
    }

    // A representative attention-backward-sized workload, just to exercise the
    // splitter end to end: batch * heads * banks * tokens.
    hipsplit::KernelProfile kp;
    kp.threads_per_block = 256;
    kp.dynamic_lds_bytes = 32 * 1024;

    hipsplit::Workload w;
    w.total_items = 8ull * 64ull * 4ull * 256ull;

    auto plan = hipsplit::Planner::plan(topo, kp, w);
    std::printf("\nplan for %llu items (%s):\n",
                static_cast<unsigned long long>(w.total_items),
                plan.note.empty() ? "ok" : plan.note.c_str());
    for (const auto& l : plan.launches) {
        std::printf("  dev %d: items [%llu, %llu)  grid=%llu  block=%d  blocks/CU=%d  sat=%.2f\n",
                    l.device_id,
                    static_cast<unsigned long long>(l.item_offset),
                    static_cast<unsigned long long>(l.item_offset + l.item_count),
                    static_cast<unsigned long long>(l.grid_size),
                    l.block_size, l.blocks_per_cu, l.saturation);
    }
    return 0;
}
