#ifndef HIPSPLIT_CLUSTER_MPI_HPP
#define HIPSPLIT_CLUSTER_MPI_HPP

// Opt-in MPI helper for discovering a cluster's topology.
//
// The planner already splits across nodes once you fill in Topology::num_nodes,
// this_node_rank and node_weights. Doing that by hand is fine, but if you are
// already running under MPI this fills those fields for you:
//
//   #include <hipsplit/cluster_mpi.hpp>
//   hipsplit::Topology topo = hipsplit::query_cluster_topology(MPI_COMM_WORLD);
//   hipsplit::LaunchPlan plan = hipsplit::Planner::plan(topo, kp, w);
//
// It is kept deliberately separate -- its own header, its own build target
// (hipsplit::mpi), gated behind -DHIPSPLIT_WITH_MPI=ON -- so neither MPI nor
// this header ever leaks into the core or the default build. Pull it in only if
// you want it.
//
// Scope, stated honestly: this *discovers topology and decides the tiling*. It
// moves no data. Exchanging activations/gradients between nodes is your
// transport's job (RCCL/MPI), because it is inseparable from your tensors.

#include "hipsplit/topology.hpp"

#include <mpi.h>

namespace hipsplit {

// Build a cluster-aware Topology for the calling rank.
//
// Queries this rank's local GPUs (query_topology), then exchanges per-rank
// compute weight across `comm` so the planner can divide work between nodes in
// proportion to their real throughput. Sets:
//   - devices         : this rank's local GPUs (unchanged)
//   - this_node_rank  : this rank's index in `comm`
//   - num_nodes       : the size of `comm`
//   - node_weights    : summed local compute_weight() of every rank, in order
//
// Assumes one MPI rank per node -- the natural shape for a node -> device split.
// A rank that sees no GPU contributes zero weight and is simply handed no work
// by the planner (its local plan will be empty); that is intentional and the
// caller should treat it as "nothing to run here".
Topology query_cluster_topology(MPI_Comm comm = MPI_COMM_WORLD);

} // namespace hipsplit

#endif // HIPSPLIT_CLUSTER_MPI_HPP
