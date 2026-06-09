#include "hipsplit/occupancy.hpp"

#include <hip/hip_runtime.h>

namespace hipsplit {

int occupancy_blocks_per_cu(const void* kernel, int block_size, std::size_t dynamic_lds) {
    int blocks = 0;
    hipError_t err = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, block_size, dynamic_lds);
    return err == hipSuccess ? blocks : 0;
}

bool suggest_block_size(const void* kernel, std::size_t dynamic_lds,
                        int block_size_limit, int* out_block_size, int* out_min_grid) {
    int grid = 0;
    int block = 0;
    hipError_t err = hipOccupancyMaxPotentialBlockSize(
        &grid, &block, kernel, dynamic_lds, block_size_limit);
    if (err != hipSuccess) return false;
    if (out_block_size) *out_block_size = block;
    if (out_min_grid)   *out_min_grid   = grid;
    return true;
}

} // namespace hipsplit
