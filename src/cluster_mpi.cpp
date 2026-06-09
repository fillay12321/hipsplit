#include "hipsplit/cluster_mpi.hpp"
#include "hipsplit/device_query.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace hipsplit {

Topology query_cluster_topology(MPI_Comm comm) {
    // Start from whatever GPUs this rank can actually see.
    Topology topo = query_topology();

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    topo.this_node_rank = rank;
    topo.num_nodes      = size;

    // This rank's share of the cluster is the combined muscle of its local GPUs.
    // A rank with no visible GPU contributes 0, so the planner gives it nothing
    // -- which is the honest outcome: it has nothing to run the work on.
    double local_weight = 0.0;
    for (const auto& d : topo.devices) {
        local_weight += d.compute_weight();
    }

    // Gather every rank's weight so all ranks agree on the same node split. The
    // planner's node-level division is deterministic given identical weights, so
    // each rank independently carves out the same contiguous global tiling.
    std::vector<double> weights(static_cast<std::size_t>(size), 0.0);
    MPI_Allgather(&local_weight, 1, MPI_DOUBLE,
                  weights.data(), 1, MPI_DOUBLE, comm);
    topo.node_weights = std::move(weights);

    return topo;
}

} // namespace hipsplit
