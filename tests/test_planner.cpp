#include "hipsplit/planner.hpp"
#include "microtest.hpp"

#include <cstdint>

using namespace hipsplit;

// A stand-in for an RDNA 3.5 part (gfx1151, 40 CUs). Hand-built so the tests
// need no GPU.
static DeviceInfo strix_halo(int id = 0) {
    DeviceInfo d;
    d.device_id            = id;
    d.name                 = "AMD Radeon 8060S";
    d.arch                 = "gfx1151";
    d.compute_units        = 40;
    d.max_threads_per_block = 1024;
    d.max_threads_per_cu   = 2048;   // ballpark: 64 wavefronts * 32 lanes
    d.wavefront_size       = 32;
    d.lds_per_block        = 65536;
    d.lds_per_cu           = 65536;
    d.max_blocks_per_cu    = 32;
    d.clock_khz            = 2900000;
    return d;
}

TEST(auto_block_size_is_wavefront_multiple) {
    DeviceInfo d = strix_halo();
    KernelProfile kp; // auto
    int bs = Planner::choose_block_size(d, kp);
    CHECK(bs % d.wavefront_size == 0);
    CHECK(bs >= d.wavefront_size);
    CHECK(bs <= d.max_threads_per_block);
}

TEST(explicit_block_size_is_clamped_and_rounded) {
    DeviceInfo d = strix_halo();
    KernelProfile kp;
    kp.threads_per_block = 9999; // over the hardware limit
    int bs = Planner::choose_block_size(d, kp);
    CHECK(bs <= d.max_threads_per_block);
    CHECK(bs % d.wavefront_size == 0);
}

TEST(huge_lds_kills_occupancy) {
    DeviceInfo d = strix_halo();
    // ask for more LDS than a single block may have -> zero resident blocks
    int bpc = Planner::estimate_blocks_per_cu(d, 256, d.lds_per_block + 1);
    CHECK_EQ(bpc, 0);
}

TEST(lds_limits_blocks_per_cu) {
    DeviceInfo d = strix_halo();
    // 32 KB per block on a 64 KB CU -> at most 2 resident blocks from LDS alone
    int bpc = Planner::estimate_blocks_per_cu(d, 256, 32 * 1024);
    CHECK(bpc <= 2);
    CHECK(bpc >= 1);
}

TEST(grid_covers_all_work) {
    DeviceInfo d = strix_halo();
    KernelProfile kp;
    kp.threads_per_block = 256;
    Workload w;
    w.total_items = 100000;
    w.items_per_thread = 1;
    DeviceLaunch dl = Planner::plan_one(d, kp, w, 0, w.total_items);
    // every item must be covered by some thread
    CHECK(dl.grid_size * static_cast<std::uint64_t>(dl.block_size) >= w.total_items);
    // but we shouldn't overshoot by more than one block
    CHECK((dl.grid_size - 1) * static_cast<std::uint64_t>(dl.block_size) < w.total_items);
}

TEST(single_gpu_plan_is_valid) {
    Topology t;
    t.devices.push_back(strix_halo());
    KernelProfile kp;
    kp.threads_per_block = 256;
    Workload w;
    w.total_items = 1u << 20;
    LaunchPlan p = Planner::plan(t, kp, w);
    CHECK(p.valid);
    CHECK_EQ(p.launches.size(), static_cast<std::size_t>(1));
    CHECK_EQ(p.launches[0].item_offset, static_cast<std::uint64_t>(0));
    CHECK_EQ(p.launches[0].item_count, w.total_items);
}

TEST(work_is_split_and_conserved_across_two_gpus) {
    Topology t;
    t.devices.push_back(strix_halo(0));
    t.devices.push_back(strix_halo(1));
    KernelProfile kp;
    kp.threads_per_block = 256;
    Workload w;
    w.total_items = 1000003; // prime-ish, forces remainder handling
    LaunchPlan p = Planner::plan(t, kp, w);
    CHECK(p.valid);
    std::uint64_t covered = 0;
    std::uint64_t expect_offset = 0;
    for (auto& l : p.launches) {
        CHECK_EQ(l.item_offset, expect_offset); // contiguous: no gaps, no overlap
        expect_offset += l.item_count;
        covered += l.item_count;
    }
    CHECK_EQ(covered, w.total_items); // nothing lost or duplicated
}

TEST(uneven_gpus_get_proportional_work) {
    Topology t;
    DeviceInfo big   = strix_halo(0);  // 40 CUs
    DeviceInfo small = strix_halo(1);
    small.compute_units = 10;          // a quarter of the CUs
    small.clock_khz     = big.clock_khz;
    t.devices.push_back(big);
    t.devices.push_back(small);
    KernelProfile kp;
    kp.threads_per_block = 256;
    Workload w;
    w.total_items = 1000000;
    LaunchPlan p = Planner::plan(t, kp, w);
    CHECK_EQ(p.launches.size(), static_cast<std::size_t>(2));
    // the big device should get roughly 4x the small one's share
    double ratio = static_cast<double>(p.launches[0].item_count) /
                   static_cast<double>(p.launches[1].item_count);
    CHECK(ratio > 3.0 && ratio < 5.0);
}

TEST(split_granularity_is_respected) {
    Topology t;
    t.devices.push_back(strix_halo(0));
    t.devices.push_back(strix_halo(1));
    KernelProfile kp;
    kp.threads_per_block = 256;
    Workload w;
    w.total_items = 8000;
    w.split_granularity = 64; // keep whole heads together
    LaunchPlan p = Planner::plan(t, kp, w);
    std::uint64_t covered = 0;
    for (auto& l : p.launches) covered += l.item_count;
    CHECK_EQ(covered, w.total_items);
}

// ---- Multi-node: the same planner, two levels of splitting ----

static Topology node_of(int rank, int num_nodes) {
    Topology t;
    t.devices.push_back(strix_halo(0));
    t.this_node_rank = rank;
    t.num_nodes      = num_nodes;
    return t;
}

TEST(single_node_default_leaves_node_slice_whole) {
    Topology t;
    t.devices.push_back(strix_halo());
    KernelProfile kp;
    kp.threads_per_block = 256;
    Workload w;
    w.total_items = 1u << 20;
    LaunchPlan p = Planner::plan(t, kp, w);
    CHECK(p.valid);
    CHECK_EQ(p.node_item_offset, static_cast<std::uint64_t>(0));
    CHECK_EQ(p.node_item_count, w.total_items);
}

TEST(cluster_split_tiles_contiguously_across_nodes) {
    KernelProfile kp;
    kp.threads_per_block = 256;
    Workload w;
    w.total_items = 1000000;

    LaunchPlan p0 = Planner::plan(node_of(0, 2), kp, w);
    LaunchPlan p1 = Planner::plan(node_of(1, 2), kp, w);
    CHECK(p0.valid);
    CHECK(p1.valid);

    // node slices are contiguous and together cover the whole workload
    CHECK_EQ(p0.node_item_offset, static_cast<std::uint64_t>(0));
    CHECK_EQ(p0.node_item_offset + p0.node_item_count, p1.node_item_offset);
    CHECK_EQ(p1.node_item_offset + p1.node_item_count, w.total_items);

    // every launch on a node stays inside that node's global slice
    for (auto& l : p0.launches) {
        CHECK(l.item_offset >= p0.node_item_offset);
        CHECK(l.item_offset + l.item_count <= p0.node_item_offset + p0.node_item_count);
    }
    for (auto& l : p1.launches) {
        CHECK(l.item_offset >= p1.node_item_offset);
        CHECK(l.item_offset + l.item_count <= p1.node_item_offset + p1.node_item_count);
    }
}

TEST(weighted_nodes_get_proportional_slices) {
    KernelProfile kp;
    kp.threads_per_block = 256;
    Workload w;
    w.total_items = 800000;

    auto weighted = [](int rank) {
        Topology t = node_of(rank, 2);
        t.node_weights = {3.0, 1.0}; // node 0 is three times node 1
        return t;
    };
    LaunchPlan p0 = Planner::plan(weighted(0), kp, w);
    LaunchPlan p1 = Planner::plan(weighted(1), kp, w);
    double ratio = static_cast<double>(p0.node_item_count) /
                   static_cast<double>(p1.node_item_count);
    CHECK(ratio > 2.5 && ratio < 3.5);
    CHECK_EQ(p0.node_item_count + p1.node_item_count, w.total_items);
}

int main() { return microtest::run_all(); }
