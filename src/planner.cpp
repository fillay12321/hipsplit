#include "hipsplit/planner.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace hipsplit {
namespace {

// Split `total_units` into per-bucket counts proportional to `weights`, giving
// any rounding remainder to the first bucket so the parts sum back to the whole.
// Deterministic: identical inputs always produce an identical partition, which
// is what lets every node agree on the same cluster-level tiling.
std::vector<std::uint64_t> split_units_by_weight(const std::vector<double>& weights,
                                                 std::uint64_t total_units) {
    std::vector<std::uint64_t> out(weights.size(), 0);
    if (weights.empty()) return out;

    double total_weight = 0.0;
    for (double w : weights) total_weight += (w > 0.0 ? w : 0.0);
    if (total_weight <= 0.0) total_weight = static_cast<double>(weights.size());

    std::uint64_t assigned = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const double wgt  = weights[i] > 0.0 ? weights[i] : 0.0;
        const double frac = wgt / total_weight;
        out[i] = static_cast<std::uint64_t>(frac * static_cast<double>(total_units));
        assigned += out[i];
    }
    if (assigned < total_units) out[0] += (total_units - assigned);
    return out;
}

} // namespace

int Planner::choose_block_size(const DeviceInfo& dev, const KernelProfile& kp) {
    const int wf     = dev.wavefront_size > 0 ? dev.wavefront_size : 64;
    const int mult   = kp.block_size_multiple > 0 ? kp.block_size_multiple : wf;
    const int hw_max = dev.max_threads_per_block > 0 ? dev.max_threads_per_block : 1024;

    auto round_to_mult = [mult](int v) {
        int r = (v / mult) * mult;
        return r < mult ? mult : r;
    };

    if (kp.threads_per_block > 0) {
        int bs = round_to_mult(kp.threads_per_block);
        if (bs > hw_max) bs = round_to_mult(hw_max);
        return bs;
    }

    // No explicit size: aim for 256 threads, the usual sweet spot on RDNA
    // (enough wavefronts to hide latency without starving each block of
    // registers), then clamp to the profile and hardware ceilings. When a real
    // kernel pointer is available, prefer the HIP occupancy API over this
    // analytical guess.
    const int limit  = std::min(kp.block_size_limit > 0 ? kp.block_size_limit : hw_max, hw_max);
    const int target = std::min(256, limit);
    return round_to_mult(target);
}

int Planner::estimate_blocks_per_cu(const DeviceInfo& dev, int block_size,
                                    std::size_t total_lds_per_block) {
    if (block_size <= 0) return 0;

    // A block that wants more shared memory than a single workgroup is allowed
    // simply cannot launch. That's a hard zero, not a small number.
    if (total_lds_per_block > 0 && dev.lds_per_block > 0 &&
        total_lds_per_block > dev.lds_per_block) {
        return 0;
    }

    int limit = dev.max_blocks_per_cu > 0 ? dev.max_blocks_per_cu : 32;

    if (dev.max_threads_per_cu > 0) {
        limit = std::min(limit, dev.max_threads_per_cu / block_size);
    }
    if (total_lds_per_block > 0) {
        const std::size_t lds_cu = dev.lds_per_cu > 0 ? dev.lds_per_cu : dev.lds_per_block;
        const int by_lds = lds_cu > 0 ? static_cast<int>(lds_cu / total_lds_per_block) : 0;
        limit = std::min(limit, by_lds);
    }
    return limit < 0 ? 0 : limit;
}

DeviceLaunch Planner::plan_one(const DeviceInfo& dev, const KernelProfile& kp,
                               const Workload& w, std::uint64_t item_offset,
                               std::uint64_t item_count) {
    DeviceLaunch dl;
    dl.device_id   = dev.device_id;
    dl.item_offset = item_offset;
    dl.item_count  = item_count;

    const int bs = choose_block_size(dev, kp);
    dl.block_size = bs;

    const int ipt = w.items_per_thread > 0 ? w.items_per_thread : 1;
    const std::uint64_t per_block = static_cast<std::uint64_t>(bs) *
                                    static_cast<std::uint64_t>(ipt);
    dl.grid_size = per_block > 0 ? (item_count + per_block - 1) / per_block : 0;

    const std::size_t lds = kp.dynamic_lds_bytes + kp.static_lds_bytes;
    dl.blocks_per_cu = estimate_blocks_per_cu(dev, bs, lds);

    const std::uint64_t keep_busy =
        static_cast<std::uint64_t>(dl.blocks_per_cu) *
        static_cast<std::uint64_t>(dev.compute_units > 0 ? dev.compute_units : 1);
    dl.saturation = keep_busy > 0
        ? static_cast<double>(dl.grid_size) / static_cast<double>(keep_busy)
        : 0.0;
    return dl;
}

LaunchPlan Planner::plan(const Topology& topo, const KernelProfile& kp,
                         const Workload& w) {
    LaunchPlan plan;
    if (topo.empty())       { plan.note = "no devices in topology"; return plan; }
    if (w.total_items == 0) { plan.note = "workload is empty";      return plan; }

    const std::uint64_t gran        = w.split_granularity > 0 ? w.split_granularity : 1;
    const std::uint64_t total_units = (w.total_items + gran - 1) / gran;

    // ---- Level 1: split across nodes (a no-op for a single-node cluster) ----
    const int num_nodes = topo.num_nodes > 0 ? topo.num_nodes : 1;
    int rank = topo.this_node_rank;
    if (rank < 0)          rank = 0;
    if (rank >= num_nodes) rank = num_nodes - 1;

    std::uint64_t node_unit_offset = 0;
    std::uint64_t node_units       = total_units;
    if (num_nodes > 1) {
        std::vector<double> node_weights;
        if (static_cast<int>(topo.node_weights.size()) == num_nodes) {
            node_weights = topo.node_weights;            // caller-supplied bias
        } else {
            node_weights.assign(static_cast<std::size_t>(num_nodes), 1.0); // homogeneous
        }
        const std::vector<std::uint64_t> per_node =
            split_units_by_weight(node_weights, total_units);
        for (int i = 0; i < rank; ++i)
            node_unit_offset += per_node[static_cast<std::size_t>(i)];
        node_units = per_node[static_cast<std::size_t>(rank)];
    }

    plan.node_item_offset = node_unit_offset * gran;
    if (plan.node_item_offset > w.total_items) plan.node_item_offset = w.total_items;
    std::uint64_t node_items = node_units * gran;
    if (plan.node_item_offset + node_items > w.total_items)
        node_items = w.total_items - plan.node_item_offset;
    plan.node_item_count = node_items;

    if (node_items == 0) {
        plan.note = "this node's slice of the cluster workload is empty";
        return plan; // valid stays false: there is nothing to launch on this node
    }

    // ---- Level 2: split this node's slice across its local devices ----
    double total_weight = 0.0;
    for (const auto& d : topo.devices) total_weight += d.compute_weight();
    if (total_weight <= 0.0) total_weight = static_cast<double>(topo.size());

    std::vector<std::uint64_t> units(topo.size(), 0);
    std::uint64_t assigned = 0;
    for (std::size_t i = 0; i < topo.size(); ++i) {
        const double frac = topo.devices[i].compute_weight() / total_weight;
        units[i] = static_cast<std::uint64_t>(frac * static_cast<double>(node_units));
        assigned += units[i];
    }
    // Rounding always leaves a few granules unassigned; give them to the first
    // (by convention strongest) device so this node's totals add up exactly.
    if (assigned < node_units) units[0] += (node_units - assigned);

    std::uint64_t offset = plan.node_item_offset;
    const std::uint64_t node_end = plan.node_item_offset + node_items;
    for (std::size_t i = 0; i < topo.size() && offset < node_end; ++i) {
        if (units[i] == 0) continue;
        std::uint64_t count = units[i] * gran;
        if (offset + count > node_end) count = node_end - offset;
        if (count == 0) continue;
        plan.launches.push_back(plan_one(topo.devices[i], kp, w, offset, count));
        offset += count;
    }

    bool starved = false;
    for (const auto& l : plan.launches) if (l.saturation < 1.0) starved = true;
    if (starved) {
        plan.note = "grid does not fully saturate at least one device; a larger "
                    "problem or items_per_thread=1 would use it better";
    }
    plan.valid = !plan.launches.empty();
    return plan;
}

} // namespace hipsplit
