#include "hipsplit/autoplan.hpp"

#include "hipsplit/device_query.hpp"

#include <cstdlib>

namespace hipsplit {

LaunchPlan plan_auto(const KernelProfile& kp, const Workload& w) {
    // query_topology() returns an empty topology when there's no GPU, and
    // Planner::plan() turns that into a clearly-invalid plan with a note, so
    // there's nothing special to handle there.
    Topology topo = query_topology();

    // Multi-node, out of the box: if the launcher advertises a cluster through
    // env vars, fold that into the topology so the planner carves out just this
    // node's slice. With these unset we plan as a single machine. We use our own
    // unambiguous variable names rather than guessing at framework-specific ones
    // (a global process rank is not the same thing as a node rank).
    if (const char* nn = std::getenv("HIPSPLIT_NUM_NODES")) {
        const int n = std::atoi(nn);
        if (n > 1) {
            topo.num_nodes = n;
            if (const char* nr = std::getenv("HIPSPLIT_NODE_RANK")) {
                topo.this_node_rank = std::atoi(nr);
            }
        }
    }

    return Planner::plan(topo, kp, w);
}

} // namespace hipsplit
