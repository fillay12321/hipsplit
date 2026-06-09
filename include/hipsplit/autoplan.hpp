#ifndef HIPSPLIT_AUTOPLAN_HPP
#define HIPSPLIT_AUTOPLAN_HPP

#include "hipsplit/planner.hpp"

namespace hipsplit {

// One-call convenience for the common case: query whatever HIP devices are
// present right now and hand back a launch plan for them. Use this when you
// just want a plan for this machine and don't care to inspect the topology
// yourself.
//
// When no HIP device is visible the returned plan has valid == false and a note
// explaining why, which leaves a clean opening for a CPU fallback.
LaunchPlan plan_auto(const KernelProfile& kp, const Workload& w);

} // namespace hipsplit

#endif // HIPSPLIT_AUTOPLAN_HPP
