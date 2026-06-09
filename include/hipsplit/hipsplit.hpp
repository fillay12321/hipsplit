#ifndef HIPSPLIT_HPP
#define HIPSPLIT_HPP

// Single include for the whole module. If you only want the GPU-free planning
// core (no HIP dependency), include topology.hpp and planner.hpp directly
// instead of this umbrella header.

#include "hipsplit/topology.hpp"
#include "hipsplit/planner.hpp"
#include "hipsplit/device_query.hpp"
#include "hipsplit/occupancy.hpp"
#include "hipsplit/autoplan.hpp"

// C ABI, for callers that reach hipsplit over the C boundary (Rust FFI, etc.).
#include "hipsplit/hipsplit_c.h"

#endif // HIPSPLIT_HPP
