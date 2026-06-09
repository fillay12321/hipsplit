#ifndef HIPSPLIT_OCCUPANCY_HPP
#define HIPSPLIT_OCCUPANCY_HPP

#include <cstddef>

namespace hipsplit {

// Ask the HIP runtime directly how many blocks of a given size will fit on a CU
// for a specific kernel. This beats the analytical estimate because it also
// accounts for register pressure, which we can't see from hipDeviceProp_t
// alone. Pass the kernel as a plain function pointer cast to const void*.
// Returns 0 if HIP can't answer.
int occupancy_blocks_per_cu(const void* kernel, int block_size, std::size_t dynamic_lds);

// Let HIP suggest a block size that maximizes occupancy for this kernel. On
// success writes the suggestion through the out-params and returns true; on
// failure returns false and leaves them untouched.
bool suggest_block_size(const void* kernel, std::size_t dynamic_lds,
                        int block_size_limit, int* out_block_size, int* out_min_grid);

} // namespace hipsplit

#endif // HIPSPLIT_OCCUPANCY_HPP
