#ifndef HIPSPLIT_DEVICE_QUERY_HPP
#define HIPSPLIT_DEVICE_QUERY_HPP

#include "hipsplit/topology.hpp"

namespace hipsplit {

// Query every visible HIP device and build a Topology. Needs a working HIP
// runtime (defined in device_query.cpp, compiled only into the HIP-backed part
// of the library). Returns an empty topology when HIP reports no devices, which
// lets a caller cleanly fall back to a CPU path.
Topology query_topology();

// Query a single device by its HIP ordinal. Throws std::runtime_error if HIP
// can't describe the device.
DeviceInfo query_device(int device_id);

} // namespace hipsplit

#endif // HIPSPLIT_DEVICE_QUERY_HPP
