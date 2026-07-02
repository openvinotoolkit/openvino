// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_collector.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#if defined(_MSC_VER)
#    include <intrin.h>
#endif

#include "graph_debug_dump.hpp"
#include "op/device_subgraph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/utils/utils.hpp"
namespace {

template <typename T>
static std::vector<T> addition(const std::vector<T>& vector1, const std::vector<T>& vector2) {
    std::vector<T> addition;
    std::copy_if(vector1.begin(), vector1.end(), std::back_inserter(addition), [&vector2](const T& arg) {
        return (std::find(vector2.begin(), vector2.end(), arg) == vector2.end());
    });
    return addition;
}

template <typename T>
static size_t get_index(const std::vector<T>& vector, const T& item) {
    const auto& it = std::find(vector.begin(), vector.end(), item);
    OPENVINO_ASSERT(it != vector.end());
    return static_cast<size_t>(std::distance(vector.begin(), it));
}

}  // namespace

std::shared_ptr<ov::Node> ov::hetero::SubgraphCollector::output_node(
    const ov::hetero::SubgraphCollector::Output& output) const {
    return output.get_node_shared_ptr();
}

std::shared_ptr<ov::Node> ov::hetero::SubgraphCollector::input_node(
    const ov::hetero::SubgraphCollector::Input& input) const {
    return output_node(input.get_source_output());
}

ov::hetero::SubgraphCollector::SubgraphCollector(const std::shared_ptr<ov::Model>& model,
                                                 const AffinitiesMap& affinities)
    : _ordered_ops{model->get_ordered_ops()},
      _origin_parameters(model->get_parameters()),
      _origin_results(model->get_results()),
      _origin_sinks(model->get_sinks()),
      _intermediate_parameters{},
      _intermediate_results(),
      _affinities{affinities},
      _subgraph_inputs{},
      _subgraph_parameter_to_prev_result{} {
    init();
    _subgraph_ids = split_cyclic_dependencies();
}

bool ov::hetero::SubgraphCollector::is_graph_input_node(const ov::Node* node) const {
    return ov::op::util::is_parameter(node) || ov::op::util::is_constant(node);
}

void ov::hetero::SubgraphCollector::init() {
    // Seed _subgraph_inputs from per-node affinities: Parameters/Constants are self-boundaries,
    // and any cross-affinity input edge is a boundary; Result nodes inherit producer affinity.
    for (const auto& node : _ordered_ops) {
        if (is_graph_input_node(node.get())) {
            _subgraph_inputs.insert(Input{node.get(), 0});
        } else {
            const auto& node_affinity = _affinities.at(node);
            for (const auto& input : node->inputs()) {
                const auto source = input_node(input);
                if (node_affinity != _affinities.at(source)) {
                    if (ov::op::util::is_output(node)) {
                        _affinities[node] = _affinities.at(source);
                    } else {
                        _subgraph_inputs.insert(input);
                    }
                }
            }
        }
    }
}

ov::hetero::SubgraphCollector::SubgraphIdsMap ov::hetero::SubgraphCollector::split_cyclic_dependencies() {
    // Iteratively detect cross-subgraph cycles (subgraph A feeds B and B feeds A) and break them
    // by promoting offending edges into _subgraph_inputs until no new boundaries are added.
    //
    // Returns the final SubgraphIdsMap (valid w.r.t. _subgraph_inputs at return), so the caller
    // does not re-run collect_subgraphs_ids(). The map is also reused across loop boundaries:
    // the per-node loop's last iteration always exits without adding boundaries (loop
    // condition), so its `subgraph_ids` is still valid for the SCC loop's first iteration; and
    // each SCC iteration only recomputes after it actually modified _subgraph_inputs.
    const size_t nodes_count = _ordered_ops.size();
    std::unordered_map<const ov::Node*, size_t> node_to_index;
    node_to_index.reserve(nodes_count);
    std::vector<InputVector> ordered_inputs(nodes_count);
    std::vector<std::vector<size_t>> output_consumer_counts(nodes_count);
    for (size_t i = 0; i < nodes_count; ++i) {
        node_to_index.emplace(_ordered_ops[i].get(), i);
        ordered_inputs[i] = _ordered_ops[i]->inputs();
        const auto outputs = _ordered_ops[i]->outputs();
        auto& consumer_counts = output_consumer_counts[i];
        consumer_counts.reserve(outputs.size());
        for (const auto& output : outputs) {
            consumer_counts.push_back(output.get_target_inputs().size());
        }
    }

    auto get_index_by_node = [&node_to_index](const ov::Node* node) {
        const auto it = node_to_index.find(node);
        OPENVINO_ASSERT(it != node_to_index.end());
        return it->second;
    };

    // Bitset helpers. Subgraph-input dependency sets are represented as packed
    // 64-bit words indexed by a dense id assigned to each member of
    // _subgraph_inputs at the start of every outer iteration. This replaces the
    // previous std::set<Input>-based propagation: union/intersect/test become
    // O(S/64) bitwise ops with no per-element hashing/comparator/allocation,
    // typically ~50-100x faster for graphs with hundreds of subgraph inputs.
    using Bits = std::vector<uint64_t>;
    auto ctz64 = [](uint64_t x) -> unsigned {
    // Precondition: x != 0. Both __builtin_ctzll(0) and _BitScanForward64 with a zero
    // mask are undefined; all call sites guard with `while (bits)` before invoking.
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
        unsigned long idx;
        _BitScanForward64(&idx, x);
        return static_cast<unsigned>(idx);
#elif defined(_MSC_VER)
        unsigned idx = 0;
        while ((x & 1ULL) == 0) {
            x >>= 1;
            ++idx;
        }
        return idx;
#else
        return static_cast<unsigned>(__builtin_ctzll(x));
#endif
    };
    auto set_bit = [](Bits& b, size_t i) {
        b[i >> 6] |= (1ULL << (i & 63));
    };
    auto bit_or = [](Bits& a, const Bits& b) {
        for (size_t i = 0; i < a.size(); ++i)
            a[i] |= b[i];
    };
    auto bit_intersects = [](const Bits& a, const Bits& b) {
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] & b[i])
                return true;
        return false;
    };
    auto bit_any = [](const Bits& a) {
        for (uint64_t v : a)
            if (v)
                return true;
        return false;
    };
    auto bit_all_of = [&](const Bits& a, const auto& pred) {
        for (size_t i = 0; i < a.size(); ++i) {
            uint64_t bits = a[i];
            while (bits) {
                const size_t b = (i << 6) + ctz64(bits);
                bits &= bits - 1;
                if (!pred(b)) {
                    return false;
                }
            }
        }
        return true;
    };

    // Subgraph-ID state is shared across the per-node loop, the SCC loop, and the return value.
    SubgraphIdsMap subgraph_ids;
    std::vector<SubgraphId> subgraph_id_by_index(nodes_count);

    // Split cyclic dependencies.
    for (size_t prev_subgraphs = 0, cyclic_split_step = 0; prev_subgraphs != _subgraph_inputs.size();
         ++cyclic_split_step) {
        OPENVINO_ASSERT(cyclic_split_step < _ordered_ops.size(), "Cannot resolve cycles during submodels split!");
        prev_subgraphs = _subgraph_inputs.size();
        subgraph_ids = collect_subgraphs_ids();

        for (const auto& node : _ordered_ops) {
            const auto index = get_index_by_node(node.get());
            subgraph_id_by_index[index] = subgraph_ids.at(node);
        }

        // === Phase 1: assign a dense bit id to every current subgraph input. ===
        const size_t S = _subgraph_inputs.size();
        const size_t W = (S + 63) / 64;
        struct InputHash {
            size_t operator()(const Input& in) const noexcept {
                // Input == {Node*, port_index}. Mix the pointer with the port to avoid
                // collisions across multiple inputs of the same node.
                const auto h1 = std::hash<const ov::Node*>{}(in.get_node());
                const auto h2 = std::hash<size_t>{}(in.get_index());
                return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
            }
        };
        std::unordered_map<Input, size_t, InputHash> input_to_bit;
        input_to_bit.reserve(S * 2);
        std::vector<Input> bit_to_input;
        bit_to_input.reserve(S);
        for (const auto& in : _subgraph_inputs) {
            input_to_bit.emplace(in, bit_to_input.size());
            bit_to_input.push_back(in);
        }

        // === Phase 2: per-bit metadata used by the per-node classification step. ===
        std::vector<SubgraphId> bit_owner_subgraph(S);  // subgraph of subgraph_input.get_node()
        std::vector<SubgraphId> bit_producer_subgraph(
            S);  // subgraph of producing node (only meaningful when not graph input)
        std::vector<uint8_t> bit_is_graph_input(S);
        for (size_t b = 0; b < S; ++b) {
            const auto& in = bit_to_input[b];
            const auto owner = in.get_node();
            bit_owner_subgraph[b] = subgraph_id_by_index[get_index_by_node(owner)];
            const bool gi = is_graph_input_node(owner);
            bit_is_graph_input[b] = gi ? 1 : 0;
            bit_producer_subgraph[b] = gi ? static_cast<SubgraphId>(-1)
                                          : subgraph_id_by_index[get_index_by_node(in.get_source_output().get_node())];
        }

        // === Phase 3: forward-propagate subgraph-input dependencies in topological order. ===
        // Equivalent to intersection(full_transitive_closure, _subgraph_inputs)
        // but bounded by |_subgraph_inputs| via the bitset width.
        std::vector<Bits> node_input_deps(nodes_count, Bits(W, 0ULL));
        for (size_t node_idx = 0; node_idx < nodes_count; ++node_idx) {
            const auto& node = _ordered_ops[node_idx];
            auto& deps = node_input_deps[node_idx];
            if (is_graph_input_node(node.get())) {
                Input self_input{node.get(), 0};
                const auto it = input_to_bit.find(self_input);
                if (it != input_to_bit.end()) {
                    set_bit(deps, it->second);
                }
            } else {
                for (const auto& input : ordered_inputs[node_idx]) {
                    const auto it = input_to_bit.find(input);
                    if (it != input_to_bit.end()) {
                        set_bit(deps, it->second);
                    }
                    const auto source_idx = get_index_by_node(input.get_source_output().get_node());
                    bit_or(deps, node_input_deps[source_idx]);
                }
            }
        }

        // === Phase 4a: classify each node's deps into same-subgraph and cyclic-feedback subsets. ===
        std::vector<Bits> node_subgraph_input_dependencies(nodes_count, Bits(W, 0ULL));
        std::vector<Bits> node_subgraph_cyclic_input_dependencies(nodes_count, Bits(W, 0ULL));
        for (size_t node_idx = 0; node_idx < nodes_count; ++node_idx) {
            const auto& deps = node_input_deps[node_idx];
            const SubgraphId my_sg = subgraph_id_by_index[node_idx];
            auto& sg_dep = node_subgraph_input_dependencies[node_idx];
            auto& cyc_dep = node_subgraph_cyclic_input_dependencies[node_idx];
            for (size_t w = 0; w < W; ++w) {
                uint64_t bits = deps[w];
                while (bits) {
                    const size_t b = (w << 6) + ctz64(bits);
                    bits &= bits - 1;
                    if (bit_owner_subgraph[b] == my_sg) {
                        set_bit(sg_dep, b);
                    }
                    if (!bit_is_graph_input[b] && bit_producer_subgraph[b] == my_sg) {
                        set_bit(cyc_dep, b);
                    }
                }
            }
        }

        // === Phase 4b: for each node with cyclic feedback, promote offending edges into _subgraph_inputs. ===
        auto promote_boundaries_for_node = [&](size_t node_idx) {
            const auto& cyc_dep = node_subgraph_cyclic_input_dependencies[node_idx];
            if (!bit_any(cyc_dep))
                return;
            const SubgraphId my_sg = subgraph_id_by_index[node_idx];
            // Collect all subgraph inputs that the cyclic feedback transitively depends on.
            Bits cyclic_inputs_dependencies(W, 0ULL);
            for (size_t w = 0; w < W; ++w) {
                uint64_t bits = cyc_dep[w];
                while (bits) {
                    const size_t b = (w << 6) + ctz64(bits);
                    bits &= bits - 1;
                    const auto& cyclic_input = bit_to_input[b];
                    const auto cyclic_input_idx = get_index_by_node(cyclic_input.get_source_output().get_node());
                    bit_or(cyclic_inputs_dependencies, node_subgraph_input_dependencies[cyclic_input_idx]);
                }
            }
            // Also include dependencies at cycle re-entry points: boundary edges where
            // data from another subgraph flows back into this one. Without this, the
            // intersection check below misses edges that bridge independently-entered
            // nodes (e.g., a shared constant) to the cycle's return path.
            const auto& sg_dep = node_subgraph_input_dependencies[node_idx];
            for (size_t w = 0; w < W; ++w) {
                uint64_t bits = sg_dep[w];
                while (bits) {
                    const size_t b = (w << 6) + ctz64(bits);
                    bits &= bits - 1;
                    if (!bit_is_graph_input[b] && bit_owner_subgraph[b] == my_sg && bit_producer_subgraph[b] != my_sg) {
                        const auto owner_idx = get_index_by_node(bit_to_input[b].get_node());
                        bit_or(cyclic_inputs_dependencies, node_subgraph_input_dependencies[owner_idx]);
                    }
                }
            }
            for (const auto& input : ordered_inputs[node_idx]) {
                const auto input_source_idx = get_index_by_node(input.get_source_output().get_node());
                const auto& src_cyc_dep = node_subgraph_cyclic_input_dependencies[input_source_idx];
                const auto& src_sg_dep = node_subgraph_input_dependencies[input_source_idx];
                if (!bit_intersects(cyc_dep, src_cyc_dep) && bit_intersects(cyclic_inputs_dependencies, src_sg_dep)) {
                    const auto source_output = input.get_source_output();
                    const bool single_consumer_graph_input_leaf =
                        output_consumer_counts[input_source_idx][source_output.get_index()] == 1 &&
                        !is_graph_input_node(source_output.get_node()) && !bit_any(src_cyc_dep) &&
                        bit_all_of(src_sg_dep, [&](size_t b) {
                            const auto& traced_input = bit_to_input[b];
                            if (is_graph_input_node(traced_input.get_node())) {
                                return true;
                            }
                            const auto* traced_producer = traced_input.get_source_output().get_node();
                            return is_graph_input_node(traced_producer);
                        });
                    if (!single_consumer_graph_input_leaf) {
                        _subgraph_inputs.insert(input);
                    }
                }
            }
        };
        for (size_t node_idx = 0; node_idx < nodes_count; ++node_idx) {
            promote_boundaries_for_node(node_idx);
        }
    }

    // === Subgraph-level SCC fallback. ===========================================================
    // The per-node heuristic above only detects cycles whose re-entry point sits on a node whose
    // own cyc_dep bitset is non-empty (same-sg data flows back through a foreign sg into that
    // node's inputs). Two classes of subgraph-DAG cycles fall outside its scope, and both are
    // first-class cases this fallback exists to handle -- neither is exceptional:
    //
    //   (a) Multi-hop subgraph-DAG cycles (sg_A -> sg_B -> sg_C -> sg_D -> sg_A) where the
    //       producer and re-entry consumer are several subgraphs apart and no single node sees
    //       its own sg on the cycle.
    //   (b) Shared-graph-input cycles, where a Constant (or other graph input) fans out to
    //       multiple consumers that Union-Find fuses into a single subgraph, and that fused
    //       subgraph then both produces and consumes data on the same neighbor subgraph. The
    //       cut edge here is an input of the foreign-sg node, not of the same-sg node whose
    //       cyc_dep is non-empty, so Phase 4b cannot promote it by construction.
    //
    // Both arise from Union-Find merging structurally independent regions via shared inputs.
    // The ov::Model itself is a DAG; the cycle is purely an artifact of subgraph fusion that
    // run()'s topological sort cannot resolve.
    //
    // Break the cycle by identifying non-trivial SCCs in the subgraph DAG and, per iteration,
    // isolating one node out of some SCC-member Union-Find component by promoting all of its
    // same-sg input edges to boundary (see isolate_one_scc_node for the rationale and the
    // convergence argument). The loop is bounded by the total number of node-input edges; in
    // practice it converges in ~#SCC iterations.

    // Helper 1: build the subgraph DAG from cross-subgraph edges already recorded in
    // _subgraph_inputs. Parallel edges between the same pair of subgraphs are de-duplicated;
    // self-edges (producer_sg == owner_sg) are filtered so single-subgraph SCCs cannot arise.
    using SgAdj = std::unordered_map<SubgraphId, std::unordered_set<SubgraphId>>;
    auto build_subgraph_adjacency =
        [&](const std::vector<SubgraphId>& sg_id_by_index) -> std::pair<SgAdj, std::unordered_set<SubgraphId>> {
        SgAdj adj;
        std::unordered_set<SubgraphId> all_sgs;
        for (size_t i = 0; i < nodes_count; ++i) {
            all_sgs.insert(sg_id_by_index[i]);
        }
        for (const auto& inp : _subgraph_inputs) {
            if (is_graph_input_node(inp.get_node()))
                continue;
            const auto owner_sg = sg_id_by_index[get_index_by_node(inp.get_node())];
            const auto producer_sg = sg_id_by_index[get_index_by_node(inp.get_source_output().get_node())];
            if (owner_sg == producer_sg)
                continue;
            adj[producer_sg].insert(owner_sg);
        }
        return {std::move(adj), std::move(all_sgs)};
    };

    // Helper 2: return the set of subgraphs that belong to any non-trivial SCC of `adj`, using
    // iterative Tarjan. An exact SCC algorithm is required here: a two-peel (forward + reverse
    // Kahn) approximation also flags acyclic bridges between two disjoint cycles (e.g. X in
    // A<->B -> X -> C<->D survives both peels), which would either waste a promotion on an
    // acyclic subgraph or trip the "no internal edge" assert below when the bridge subgraph has
    // no same-sg edge. The loop is iterative to avoid recursion depth issues on large partitions.
    auto find_non_trivial_scc_members =
        [](const SgAdj& adj, const std::unordered_set<SubgraphId>& all_sgs) -> std::unordered_set<SubgraphId> {
        std::unordered_set<SubgraphId> scc_members;
        std::unordered_map<SubgraphId, int> index_of;
        std::unordered_map<SubgraphId, int> lowlink;
        std::unordered_set<SubgraphId> on_stack;
        std::vector<SubgraphId> tarjan_stack;
        int next_index = 0;
        struct Frame {
            SubgraphId v;
            std::vector<SubgraphId> neighbors;
            size_t next_neighbor;
        };
        std::vector<Frame> call_stack;
        auto neighbors_of = [&adj](SubgraphId v) {
            std::vector<SubgraphId> out;
            const auto it = adj.find(v);
            if (it != adj.end())
                out.assign(it->second.begin(), it->second.end());
            return out;
        };
        auto open_node = [&](SubgraphId v) {
            index_of[v] = next_index;
            lowlink[v] = next_index;
            ++next_index;
            tarjan_stack.push_back(v);
            on_stack.insert(v);
            call_stack.push_back({v, neighbors_of(v), 0});
        };
        for (auto start : all_sgs) {
            if (index_of.count(start))
                continue;
            open_node(start);
            while (!call_stack.empty()) {
                auto& frame = call_stack.back();
                if (frame.next_neighbor < frame.neighbors.size()) {
                    const auto w = frame.neighbors[frame.next_neighbor++];
                    if (!index_of.count(w)) {
                        open_node(w);
                    } else if (on_stack.count(w)) {
                        lowlink[frame.v] = std::min(lowlink[frame.v], index_of[w]);
                    }
                } else {
                    const auto v = frame.v;
                    if (lowlink[v] == index_of[v]) {
                        std::vector<SubgraphId> comp;
                        while (true) {
                            const auto w = tarjan_stack.back();
                            tarjan_stack.pop_back();
                            on_stack.erase(w);
                            comp.push_back(w);
                            if (w == v)
                                break;
                        }
                        // Only non-trivial SCCs (size > 1) represent real cycles in the subgraph
                        // DAG; singletons are reported by Tarjan even for nodes with no cycle and
                        // must be excluded. Self-loops were filtered out by build_subgraph_adjacency.
                        if (comp.size() > 1) {
                            for (auto m : comp)
                                scc_members.insert(m);
                        }
                    }
                    const auto finished = frame.v;
                    call_stack.pop_back();
                    if (!call_stack.empty()) {
                        lowlink[call_stack.back().v] = std::min(lowlink[call_stack.back().v], lowlink[finished]);
                    }
                }
            }
        }
        return scc_members;
    };

    // Helper 3: isolate one Union-Find node from its SCC member by promoting ALL its
    // same-subgraph non-boundary input edges into _subgraph_inputs. Returns the number of
    // edges promoted (1 .. node's input arity).
    //
    // Rationale (why this works and the simpler alternatives don't):
    //   * Promoting a single same-sg input edge per iteration diverges: the chosen node still
    //     re-merges into the SCC via its OTHER same-sg inputs in the next collect_subgraphs_ids
    //     round, and "first-input-wins" union-find keeps it in the same component. Observed on
    //     yolo26s-seg: SCC member count grew 4 -> 26 across iterations.
    //   * Promoting only edges at entry/exit points of SCC members misses the common
    //     "shared-Constant fuses regions" case: S = {c_shared, a, b, c, ...} where c_shared is
    //     a Constant unioning multiple consumers. c_shared has no same-sg consumers in OTHER
    //     SCC members (its consumers are all in S), so it is neither an entry nor an exit, and
    //     the only same-sg input that would break the cycle — (a <- c_shared) — is skipped.
    //   * Dissolving a whole SCC-member subgraph at once explodes the partition. On
    //     yolo26s-seg the GPU mainland S has 428 nodes / 449 internal edges; full dissolution
    //     produces ~450 subgraphs and breaks downstream compile_model.
    //
    // The "isolate one node" cut is the minimum needed: by promoting all of n's same-sg
    // inputs, n becomes a Union-Find root on the next round, severed from every upstream node
    // in S (including shared-Constant connectors). Each iteration thus strictly reduces the
    // size of some SCC member by 1 (n moves to its own singleton component).
    //
    // Convergence:
    //   * In any non-trivial SCC (size > 1) of the subgraph DAG, at least one member is not a
    //     Union-Find singleton: if ALL members were singletons, the SCC-DAG cycle
    //     sg_X1 -> ... -> sg_Xk -> sg_X1 would unfold into a node-level cycle
    //     x1 -> ... -> xk -> x1 in the original ov::Model, which is a DAG.
    //   * A non-singleton Union-Find component of size m has exactly m-1 unification edges,
    //     i.e. m-1 non-boundary input edges, so at least one node in it has a same-sg input.
    //   * Each iteration isolates one such node, strictly reducing the total non-singleton
    //     mass of SCC members. The loop therefore terminates in at most nodes_count iterations
    //     and well within the total_node_inputs edge budget.
    //
    // Target selection: among all candidate nodes (in any SCC member with >= 1 same-sg input),
    // prefer cuts at actual SCC re-entry nodes and shared connectors. Falling back to the node
    // with the fewest same-sg inputs is still valid for convergence, but doing so too early may
    // peel ordinary linear compute nodes out of the main device region and create tiny
    // Parameter->op->Result submodels. Those are especially expensive for GPU compilation.
    auto isolate_one_scc_node = [&](const std::vector<SubgraphId>& sg_id_by_index,
                                    const std::unordered_set<SubgraphId>& scc_members) -> size_t {
        struct CandidateRank {
            size_t lacks_scc_boundary_input = 1;
            size_t lacks_shared_same_sg_source = 1;
            size_t has_trivial_leaf_input = 1;
            size_t is_linear_compute_node = 1;
            size_t same_sg_inputs = 0;
            size_t node_idx = 0;
        };

        auto is_better_rank = [](const CandidateRank& lhs, const CandidateRank& rhs) {
            if (lhs.lacks_scc_boundary_input != rhs.lacks_scc_boundary_input)
                return lhs.lacks_scc_boundary_input < rhs.lacks_scc_boundary_input;
            if (lhs.lacks_shared_same_sg_source != rhs.lacks_shared_same_sg_source)
                return lhs.lacks_shared_same_sg_source < rhs.lacks_shared_same_sg_source;
            if (lhs.has_trivial_leaf_input != rhs.has_trivial_leaf_input)
                return lhs.has_trivial_leaf_input < rhs.has_trivial_leaf_input;
            if (lhs.is_linear_compute_node != rhs.is_linear_compute_node)
                return lhs.is_linear_compute_node < rhs.is_linear_compute_node;
            if (lhs.same_sg_inputs != rhs.same_sg_inputs)
                return lhs.same_sg_inputs < rhs.same_sg_inputs;
            return lhs.node_idx < rhs.node_idx;
        };

        auto count_non_result_consumers = [](const std::shared_ptr<ov::Node>& node) {
            size_t non_result_consumers = 0;
            for (const auto& output : node->outputs()) {
                for (const auto& target_input : output.get_target_inputs()) {
                    if (!ov::op::util::is_output(target_input.get_node())) {
                        ++non_result_consumers;
                    }
                }
            }
            return non_result_consumers;
        };
        std::vector<size_t> non_result_consumer_counts(nodes_count, static_cast<size_t>(-1));
        auto count_non_result_consumers_by_index = [&](size_t node_idx) {
            auto& cached = non_result_consumer_counts[node_idx];
            if (cached == static_cast<size_t>(-1)) {
                cached = count_non_result_consumers(_ordered_ops[node_idx]);
            }
            return cached;
        };

        bool have_target = false;
        size_t target_idx = 0;
        CandidateRank target_rank;
        auto is_graph_input_leaf_source = [&](size_t node_idx) {
            const auto& node = _ordered_ops[node_idx];
            if (is_graph_input_node(node.get()))
                return false;

            if (count_non_result_consumers_by_index(node_idx) != 1)
                return false;

            for (const auto& input : ordered_inputs[node_idx]) {
                if (!is_graph_input_node(input.get_source_output().get_node()))
                    return false;
            }
            return true;
        };
        for (size_t i = 0; i < nodes_count; ++i) {
            const auto my_sg = sg_id_by_index[i];
            if (!scc_members.count(my_sg))
                continue;
            size_t same_sg_inputs = 0;
            bool has_scc_boundary_input = false;
            bool has_shared_same_sg_source = false;
            bool has_trivial_leaf_input = false;
            for (const auto& input : ordered_inputs[i]) {
                if (_subgraph_inputs.count(input)) {
                    if (!is_graph_input_node(input.get_node())) {
                        const auto src_idx = get_index_by_node(input.get_source_output().get_node());
                        const auto producer_sg = sg_id_by_index[src_idx];
                        has_scc_boundary_input =
                            has_scc_boundary_input || (producer_sg != my_sg && scc_members.count(producer_sg));
                    }
                    continue;
                }
                const auto src_idx = get_index_by_node(input.get_source_output().get_node());
                if (sg_id_by_index[src_idx] != my_sg)
                    continue;
                ++same_sg_inputs;
                has_shared_same_sg_source =
                    has_shared_same_sg_source || count_non_result_consumers_by_index(src_idx) > 1;
                has_trivial_leaf_input = has_trivial_leaf_input || is_graph_input_leaf_source(src_idx);
            }
            if (same_sg_inputs == 0)
                continue;

            const CandidateRank candidate_rank{
                has_scc_boundary_input ? 0UL : 1UL,
                has_shared_same_sg_source ? 0UL : 1UL,
                has_trivial_leaf_input ? 1UL : 0UL,
                (same_sg_inputs == 1 && count_non_result_consumers_by_index(i) <= 1) ? 1UL : 0UL,
                same_sg_inputs,
                i};
            const bool better_target = !have_target || is_better_rank(candidate_rank, target_rank);
            if (better_target) {
                have_target = true;
                target_idx = i;
                target_rank = candidate_rank;
            }
        }
        OPENVINO_ASSERT(have_target,
                        "Subgraph SCC fallback found a cyclic subgraph DAG but every node in "
                        "every SCC member is a Union-Find singleton; that would require a "
                        "node-level cycle in the original ov::Model, which is impossible on a DAG.");

        size_t promoted = 0;
        const auto target_sg = sg_id_by_index[target_idx];
        for (const auto& input : ordered_inputs[target_idx]) {
            if (_subgraph_inputs.count(input))
                continue;
            const auto src_idx = get_index_by_node(input.get_source_output().get_node());
            if (sg_id_by_index[src_idx] != target_sg)
                continue;
            _subgraph_inputs.insert(input);
            ++promoted;
        }
        return promoted;
    };

    size_t total_node_inputs = 0;
    for (size_t i = 0; i < nodes_count; ++i) {
        total_node_inputs += ordered_inputs[i].size();
    }
    // subgraph_ids / subgraph_id_by_index reach this point already valid w.r.t. the current
    // _subgraph_inputs: the per-node loop exits only when its last iteration adds no boundaries,
    // so the ids it computed at the top of that final iteration are still in sync. Recompute
    // only after the SCC step actually modifies _subgraph_inputs.
    bool ids_valid = true;
    for (size_t scc_step = 0;; ++scc_step) {
        OPENVINO_ASSERT(scc_step < total_node_inputs + 1,
                        "Subgraph SCC fallback did not converge: exceeded node-input edge budget");
        if (!ids_valid) {
            subgraph_ids = collect_subgraphs_ids();
            for (size_t i = 0; i < nodes_count; ++i) {
                subgraph_id_by_index[i] = subgraph_ids.at(_ordered_ops[i]);
            }
            ids_valid = true;
        }
        const size_t inputs_before_step = _subgraph_inputs.size();

        const auto sg_graph = build_subgraph_adjacency(subgraph_id_by_index);
        const auto& sg_adj = sg_graph.first;
        const auto& all_sgs = sg_graph.second;
        const auto scc_members = find_non_trivial_scc_members(sg_adj, all_sgs);
        if (scc_members.empty()) {
            break;  // subgraph DAG is acyclic, fix-point reached.
        }

        // Isolate one Union-Find node from any SCC member by promoting ALL its same-sg input
        // edges. See isolate_one_scc_node for why a single-edge cut diverges, why entry/exit
        // cuts miss shared-Constant SCCs, and the convergence argument (the candidate always
        // exists because singleton-only SCCs are impossible on a DAG).
        const size_t promoted = isolate_one_scc_node(subgraph_id_by_index, scc_members);
        OPENVINO_ASSERT(promoted > 0,
                        "Subgraph SCC fallback found a cyclic subgraph DAG but the chosen node "
                        "had no same-subgraph inputs to promote; helper invariant violated.");
        // Defensive: each iteration must grow _subgraph_inputs strictly. If insert() ever found
        // all promoted edges already present (logic bug), surface it here instead of looping
        // silently until the edge budget runs out.
        OPENVINO_ASSERT(_subgraph_inputs.size() > inputs_before_step,
                        "Subgraph SCC fallback promoted edges but _subgraph_inputs did not grow");
        ids_valid = false;  // _subgraph_inputs grew; next iteration must rebuild ids.
    }

    // Edge case: if init() produced no _subgraph_inputs at all, the per-node loop never ran and
    // subgraph_ids is empty. Materialize the final mapping in that case.
    if (subgraph_ids.empty()) {
        subgraph_ids = collect_subgraphs_ids();
    }
    return subgraph_ids;
}

ov::hetero::SubgraphCollector::SubgraphIdsMap ov::hetero::SubgraphCollector::collect_subgraphs_ids() {
    // Assign a SubgraphId to every node via Union-Find: nodes connected by non-boundary edges
    // share an id. IDs are allocated as a counter at root creation, and merges keep the first
    // input's id (matches the original numbering for backward-compatible mapping/tests).
    const size_t n = _ordered_ops.size();
    std::unordered_map<const ov::Node*, size_t> idx;
    idx.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        idx.emplace(_ordered_ops[i].get(), i);
    }

    std::vector<size_t> parent(n);
    std::iota(parent.begin(), parent.end(), size_t{0});
    std::vector<SubgraphId> comp_id(n, -1);
    auto find = [&parent](size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    // Always parent `b`'s root under `a`'s root, so `a`'s root stays canonical
    // and its comp_id (the first input's id) wins.
    auto unite_keep_first = [&](size_t a, size_t b) {
        const auto ra = find(a);
        const auto rb = find(b);
        if (ra != rb) {
            parent[rb] = ra;
        }
    };

    SubgraphId counter = 0;
    InputVector srcs_buf;
    for (size_t i = 0; i < n; ++i) {
        srcs_buf.clear();
        for (const auto& input : _ordered_ops[i]->inputs()) {
            if (_subgraph_inputs.find(input) == _subgraph_inputs.end()) {
                srcs_buf.emplace_back(input);
            }
        }
        if (srcs_buf.empty()) {
            comp_id[find(i)] = counter++;
        } else {
            const auto first_src_it = idx.find(srcs_buf.front().get_source_output().get_node());
            OPENVINO_ASSERT(first_src_it != idx.end());
            const size_t first_src = first_src_it->second;
            const SubgraphId first_id = comp_id[find(first_src)];
            unite_keep_first(first_src, i);
            for (size_t k = 1; k < srcs_buf.size(); ++k) {
                const auto src_it = idx.find(srcs_buf[k].get_source_output().get_node());
                OPENVINO_ASSERT(src_it != idx.end());
                unite_keep_first(first_src, src_it->second);
            }
            // first_src's root is preserved by unite_keep_first; reassert its id.
            comp_id[find(first_src)] = first_id;
        }
    }

    SubgraphIdsMap result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const auto id = comp_id[find(i)];
        OPENVINO_ASSERT(id != static_cast<SubgraphId>(-1),
                        "SubgraphCollector: node '",
                        _ordered_ops[i]->get_friendly_name(),
                        "' was not assigned a subgraph id (Union-Find invariant violated).");
        result.emplace(_ordered_ops[i], id);
    }
    return result;
}

void ov::hetero::SubgraphCollector::split_subgraphs_by_parameter_results() {
    // Sort subgraph inputs by order
    InputVector ordered_subgraph_inputs;
    for (auto& op : _ordered_ops) {
        for (auto& input : op->inputs()) {
            if (_subgraph_inputs.count(input)) {
                ordered_subgraph_inputs.push_back(input);
            }
        }
    }
    // Collect subgraph ordered subgraph outputs
    OutputVector ordered_subgraph_outputs;
    for (const auto& input : ordered_subgraph_inputs) {
        if (!is_graph_input_node(input.get_node())) {
            auto input_source_output = input.get_source_output();
            if (!ov::op::util::is_constant(input_source_output.get_node())) {
                ordered_subgraph_outputs.push_back(input_source_output);
            }
        }
    }
    // Break graph using insertion of result parameter split
    for (const auto& output : ordered_subgraph_outputs) {
        auto output_subgraph_id = _subgraph_ids.at(output.get_node_shared_ptr());
        auto inputs = output.get_target_inputs();
        // Collect input subsets from other subgraphs. Each subset of inputs belongs to the same subgraph
        std::map<SubgraphId, InputSet> input_subsets;
        for (const auto& input : inputs) {
            auto input_subgraph_id = _subgraph_ids.at(input.get_node()->shared_from_this());
            if (output_subgraph_id != input_subgraph_id) {
                input_subsets[input_subgraph_id].emplace(input);
            }
        }
        if (input_subsets.size()) {
            // Avoid duplicate results on the same output port
            auto result = std::make_shared<ov::op::v0::Result>(output);
            ov::copy_runtime_info(output.get_node_shared_ptr(), result);
            _subgraph_ids.emplace(result, output_subgraph_id);
            _intermediate_results.push_back(result);
            for (const auto& input_subset : input_subsets) {
                const auto& input_subgraph_id = input_subset.first;
                const auto& inputs = input_subset.second;
                // Avoid duplicate parameters in the same subgraph
                auto parameter =
                    std::make_shared<ov::op::v0::Parameter>(output.get_element_type(), output.get_partial_shape());
                _intermediate_parameters.push_back(parameter);
                for (const auto& input : inputs) {
                    output.remove_target_input(input);
                    ov::copy_runtime_info(input.get_node()->shared_from_this(), parameter);
                    input.replace_source_output(parameter->output(0));
                    _subgraph_ids.emplace(parameter, input_subgraph_id);
                    _subgraph_parameter_to_prev_result.emplace(parameter, result);
                }
            }
        }
    }
}

std::pair<ov::hetero::SubgraphsVector, ov::hetero::SubgraphsMappingInfo> ov::hetero::SubgraphCollector::run() {
    // Break graph using insertion of result parameter split
    split_subgraphs_by_parameter_results();

    // Extracts subgraph parameters, results and affinities
    auto subgraphs = collect_subgraphs();

    // Subgraph topological sort
    SubgraphsVector all_subgraphs;
    for (const auto& subgraph : subgraphs) {
        all_subgraphs.emplace_back(std::move(subgraph.second));
    }

    SubgraphsVector ordered_subgraphs;
    using NodeSet = std::unordered_set<std::shared_ptr<ov::Node>>;
    NodeSet prev_results;
    size_t subgraph_topo_sorts_step = 0;
    do {
        OPENVINO_ASSERT(subgraph_topo_sorts_step < subgraphs.size(), "Cannot sort subgraphs!");
        ++subgraph_topo_sorts_step;
        SubgraphsVector new_ordered_subgraphs;
        auto is_ordered_subgraph = [&](const Subgraph& subgraph) {
            auto& parameters = subgraph._parameters;
            return std::all_of(parameters.begin(),
                               parameters.end(),
                               [&](const ov::ParameterVector::value_type& parameter) {
                                   return (ov::util::contains(_origin_parameters, parameter) ||
                                           prev_results.count(_subgraph_parameter_to_prev_result[parameter]));
                               });
        };
        std::remove_copy_if(std::begin(all_subgraphs),
                            std::end(all_subgraphs),
                            std::back_inserter(new_ordered_subgraphs),
                            [&](const Subgraph& subgraph) {
                                return !is_ordered_subgraph(subgraph);
                            });
        all_subgraphs.erase(std::remove_if(std::begin(all_subgraphs), std::end(all_subgraphs), is_ordered_subgraph),
                            std::end(all_subgraphs));
        for (const auto& subgraph : new_ordered_subgraphs) {
            for (const auto& result : subgraph._results) {
                prev_results.insert(result);
            }
        }
        std::move(std::begin(new_ordered_subgraphs),
                  std::end(new_ordered_subgraphs),
                  std::back_inserter(ordered_subgraphs));
    } while (!all_subgraphs.empty());

    // Get submodels mapping information
    auto mapping_info = get_subgraphs_mapping_info(ordered_subgraphs);

    return {ordered_subgraphs, mapping_info};
}

std::unordered_map<ov::hetero::SubgraphCollector::SubgraphId, ov::hetero::Subgraph>
ov::hetero::SubgraphCollector::collect_subgraphs() {
    std::unordered_map<SubgraphId, Subgraph> subgraphs;
    auto update_subgraph = [&](SubgraphId subgraph_id, const std::shared_ptr<ov::Node>& node) {
        auto& subgraph = subgraphs[subgraph_id];
        auto update_affinity = [&](const std::shared_ptr<ov::Node>& node) {
            auto it_affinity = _affinities.find(node);
            if (it_affinity != _affinities.end()) {
                subgraph._affinity = it_affinity->second;
            }
        };
        if (ov::op::util::is_output(node)) {
            subgraph._results.emplace_back(ov::as_type_ptr<ov::op::v0::Result>(node));
            update_affinity(input_node(node->input(0)));
        } else if (ov::op::util::is_parameter(node)) {
            subgraph._parameters.emplace_back(ov::as_type_ptr<ov::op::v0::Parameter>(node));
            update_affinity(output_node(node->output(0)));
        } else if (ov::op::util::is_sink(node)) {
            subgraph._sinks.emplace_back(ov::as_type_ptr<ov::op::Sink>(node));
            update_affinity(input_node(node->input(0)));
        }
    };
    // Update subgraph parameters
    for (auto& op_vec : {_origin_parameters, _intermediate_parameters}) {
        for (const auto& op : op_vec) {
            const auto node = std::dynamic_pointer_cast<ov::Node>(op);
            auto subgraph_id = _subgraph_ids.at(node);
            update_subgraph(subgraph_id, node);
        }
    }
    // Update subgraph results
    for (auto& op_vec : {_origin_results, _intermediate_results}) {
        for (const auto& op : op_vec) {
            const auto node = std::dynamic_pointer_cast<ov::Node>(op);
            auto subgraph_id = _subgraph_ids.at(node);
            update_subgraph(subgraph_id, node);
        }
    }
    // Update subgraph sinks
    for (const auto& op : _origin_sinks) {
        const auto node = std::dynamic_pointer_cast<ov::Node>(op);
        auto subgraph_id = _subgraph_ids.at(node);
        update_subgraph(subgraph_id, node);
    }
    return subgraphs;
}

ov::hetero::SubgraphsMappingInfo ov::hetero::SubgraphCollector::get_subgraphs_mapping_info(
    const SubgraphsVector& ordered_subgraphs) {
    SubgraphsMappingInfo info;
    // Prepare mapping between original inputs/outputs and compiled
    // submodels inputs/outputs. Example:
    // original input 0 -> submodel 0 input 0,
    // original input 1 -> submodel 1 input 0,
    // original output 0 -> submodel 1 output 0.
    //
    // Mapping is required only because before compilation
    // submodel may be preprocessed (if legacy API used),
    // so original inputs/outputs != submodels inputs/outputs
    info._inputs_to_submodels_inputs.resize(_origin_parameters.size());
    info._outputs_to_submodels_outputs.resize(_origin_results.size());
    for (size_t id = 0; id < ordered_subgraphs.size(); id++) {
        for (size_t i = 0; i < ordered_subgraphs[id]._parameters.size(); i++) {
            for (size_t j = 0; j < _origin_parameters.size(); j++)
                if (ordered_subgraphs[id]._parameters[i] == _origin_parameters[j])
                    info._inputs_to_submodels_inputs[j] = {id, i};
        }
        for (size_t i = 0; i < ordered_subgraphs[id]._results.size(); i++) {
            for (size_t j = 0; j < _origin_results.size(); j++)
                if (ordered_subgraphs[id]._results[i] == _origin_results[j])
                    info._outputs_to_submodels_outputs[j] = {id, i};
        }
    }
    // Prepare mapping between manually splitted inputs/outputs
    // to connect tensors between compiled submodels
    for (const auto& kvp : _subgraph_parameter_to_prev_result) {
        const auto& intermed_output = ov::as_type_ptr<ov::op::v0::Result>(kvp.second);
        const auto& intermed_input = ov::as_type_ptr<ov::op::v0::Parameter>(kvp.first);
        for (size_t out_subgraph_index = 0; out_subgraph_index < ordered_subgraphs.size(); out_subgraph_index++) {
            if (ov::util::contains(ordered_subgraphs[out_subgraph_index]._results, intermed_output)) {
                for (size_t in_subgraph_index = 0; in_subgraph_index < ordered_subgraphs.size(); in_subgraph_index++) {
                    if (in_subgraph_index == out_subgraph_index)
                        continue;
                    if (ov::util::contains(ordered_subgraphs[in_subgraph_index]._parameters, intermed_input)) {
                        auto out_idx = get_index(ordered_subgraphs[out_subgraph_index]._results, intermed_output);
                        auto in_idx = get_index(ordered_subgraphs[in_subgraph_index]._parameters, intermed_input);
                        info._submodels_input_to_prev_output[{in_subgraph_index, in_idx}] = {out_subgraph_index,
                                                                                             out_idx};
                    }
                }
            }
        }
    }
    return info;
}

void ov::hetero::merge_submodels(std::vector<std::shared_ptr<ov::Model>>& submodels,
                                 const std::map<NodeInfo, NodeInfo>& submodels_input_to_prev_output) {
    // Results which should not be present in final graph
    std::set<std::string> result_names_to_be_removed;
    // Remap port indexes to names, because order of them will be modified during merge
    std::map<std::pair<size_t, std::string>, std::pair<size_t, std::string>> input_to_prev_output;
    for (const auto& kvp : submodels_input_to_prev_output) {
        const auto& input_node = submodels[kvp.first.first]->inputs()[kvp.first.second].get_node();
        const auto& output_node = submodels[kvp.second.first]->outputs()[kvp.second.second].get_node();
        input_to_prev_output[{kvp.first.first, input_node->get_friendly_name()}] = {kvp.second.first,
                                                                                    output_node->get_friendly_name()};
        result_names_to_be_removed.insert(output_node->get_friendly_name());
    }
    int submodel_in_index = static_cast<int>(submodels.size()) - 1;
    while (submodel_in_index >= 0 && input_to_prev_output.size() > 0) {
        auto& submodel_in = submodels[submodel_in_index];
        size_t port_in_index = 0;
        while (port_in_index < submodel_in->get_parameters().size()) {
            auto parameter_to_replace = submodel_in->get_parameters()[port_in_index];
            auto item = input_to_prev_output.find({submodel_in_index, parameter_to_replace->get_friendly_name()});
            if (item == input_to_prev_output.end()) {
                port_in_index++;
                continue;
            }
            const auto& submodel_out_index = item->second.first;
            const auto& submodel_out_result_name = item->second.second;
            const auto& submodel_out = submodels.at(submodel_out_index);

            std::shared_ptr<ov::op::v0::Result> result_to_replace = nullptr;
            for (auto& result : submodel_out->get_results()) {
                if (result->get_friendly_name() == submodel_out_result_name) {
                    result_to_replace = result;
                }
            }
            OPENVINO_ASSERT(result_to_replace != nullptr);
            // Get all results from previous subgraph except already existed in next subgraph
            auto add_results = addition(submodel_out->get_results(), submodel_in->get_results());

            // Get all sinks from previous subgraph except already existed in next subgraph
            auto add_sinks = addition(submodel_out->get_sinks(), submodel_in->get_sinks());

            // Get all parameters from previous subgraph except already existed in next subgraph
            auto add_parameters = addition(submodel_out->get_parameters(), submodel_in->get_parameters());

            // Reconnect appropariate target inputs to the new source output
            auto result_source = result_to_replace->get_input_source_output(0);
            auto parameter_targets = parameter_to_replace->get_output_target_inputs(0);
            for (auto parameter_target : parameter_targets) {
                parameter_target.replace_source_output(result_source);
            }

            // Update parameter and results
            submodel_in->remove_parameter(parameter_to_replace);
            submodel_in->add_parameters(add_parameters);
            submodel_in->add_results(add_results);
            submodel_in->add_sinks(add_sinks);

            // Remove processed connection
            input_to_prev_output.erase(item);

            // Update incoming model since it is merged
            for (size_t i = 0; i < submodels.size(); i++) {
                if (submodels[i] == submodel_out) {
                    submodels[i] = submodel_in;
                }
            }

            // Start check ports from the beginning because number of ports are modified
            port_in_index = 0;
        }
        --submodel_in_index;
    }
    // Finally all subgraphs should be merged into single one
    OPENVINO_ASSERT(input_to_prev_output.size() == 0);
    std::set<size_t> distinct_submodels_index;
    for (size_t i = 0; i < submodels.size(); i++) {
        bool has_same_model = false;
        for (auto& index : distinct_submodels_index) {
            if (submodels[i] == submodels[index]) {
                has_same_model = true;
                break;
            }
        }
        if (!has_same_model) {
            distinct_submodels_index.insert(i);
        }
    }
    auto& result_model = submodels[0];
    for (size_t i = 1; i < submodels.size(); i++) {
        if (submodels[i] != result_model && distinct_submodels_index.count(i)) {
            result_model->add_parameters(submodels[i]->get_parameters());
            result_model->add_results(submodels[i]->get_results());
            result_model->add_sinks(submodels[i]->get_sinks());
        }
        submodels[i] = result_model;
    }
    OPENVINO_ASSERT(all_of(submodels.begin(), submodels.end(), [&](const std::shared_ptr<ov::Model>& submodel) {
        return submodel == result_model;
    }));
    // Cleanup intermidiate results
    for (size_t i = 0; i < result_model->get_results().size();) {
        auto& result = result_model->get_results()[i];
        if (result_names_to_be_removed.count(result->get_friendly_name())) {
            result_model->remove_result(result);
        } else {
            i++;
        }
    }
    submodels.resize(1);
}

std::pair<ov::hetero::SubgraphsVector, ov::hetero::SubgraphsMappingInfo> ov::hetero::get_model_subgraphs(
    const std::shared_ptr<ov::Model>& model,
    ov::SupportedOpsMap& supported_ops,
    const bool user_set_affinities,
    const bool dump_dot_files,
    const std::string default_device) {
    std::unordered_set<std::string> devices;
    ov::hetero::SubgraphCollector::AffinitiesMap affinities;
    ov::SupportedOpsMap debug_supported_ops{supported_ops};
    // Check that all nodes has user or plugin defined affinitie
    std::function<void(const std::shared_ptr<ov::Model>&, const std::string&)> collect_affinities =
        [&](const std::shared_ptr<ov::Model>& model, const std::string& default_device) {
            for (const auto& node : model->get_ordered_ops()) {
                auto it_affinity = supported_ops.find(node->get_friendly_name());
                if (it_affinity != supported_ops.end()) {
                    affinities[node] = it_affinity->second;
                    devices.emplace(it_affinity->second);
                } else if (!default_device.empty()) {
                    affinities[node] = default_device;
                    devices.emplace(default_device);
                    debug_supported_ops.insert({node->get_friendly_name(), default_device});
                } else if (!user_set_affinities) {
                    OPENVINO_THROW("Hetero device used default fallback policy, but some layers eg: \n(Name:",
                                   node->get_friendly_name(),
                                   ", Type: ",
                                   node->get_type_name(),
                                   ") were not able to be assigned on any pointed device.\n",
                                   "It happened because these layers are not supported in plugins by default.\n",
                                   "You need to implement custom layers to support them.");
                } else {
                    OPENVINO_THROW(
                        "Model passed to CompiledModel has affinity assigned, but some layers eg: \n(Name:",
                        node->get_friendly_name(),
                        ", Type: ",
                        node->get_type_name(),
                        ") were not assigned to any device.\n",
                        "It might happen if you assigned layers manually and missed some layers or\n",
                        "if you used some automatic assigning mode which decided that these layers are not\n",
                        "supported by any plugin");
                }
                if (dump_dot_files) {
                    if (auto multi_subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
                        for (size_t i = 0; i < multi_subgraph_op->get_internal_subgraphs_size(); ++i) {
                            if (const auto& sub_graph = multi_subgraph_op->get_function(i)) {
                                collect_affinities(sub_graph, debug_supported_ops.at(node->get_friendly_name()));
                            }
                        }
                    }
                }
            }
        };
    collect_affinities(model, default_device);
    if (dump_dot_files) {
        ov::hetero::debug::dump_affinities(model, debug_supported_ops, devices);
    }

    // Init subgraph collector
    ov::hetero::SubgraphCollector subgraph_collector(model, affinities);

    if (dump_dot_files) {
        auto subgraph_ids = subgraph_collector.get_subgraph_ids();
        std::map<std::string, ov::hetero::SubgraphCollector::SubgraphId> map_id;
        std::function<void(const std::shared_ptr<ov::Model>&, const ov::hetero::SubgraphCollector::SubgraphId&)>
            collect_map_id = [&](const std::shared_ptr<ov::Model>& model,
                                 const ov::hetero::SubgraphCollector::SubgraphId& default_id) {
                for (const auto& node : model->get_ordered_ops()) {
                    ov::hetero::SubgraphCollector::SubgraphId subgraph_id;
                    if (subgraph_ids.count(node)) {
                        subgraph_id = subgraph_ids.at(node);
                    } else {
                        OPENVINO_ASSERT(default_id >= 0, "Invalid default id for node " + node->get_friendly_name());
                        subgraph_id = default_id;
                    }
                    map_id.emplace(node->get_friendly_name(), subgraph_id);
                    if (auto multi_subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
                        for (size_t i = 0; i < multi_subgraph_op->get_internal_subgraphs_size(); ++i) {
                            if (const auto& sub_graph = multi_subgraph_op->get_function(i)) {
                                collect_map_id(sub_graph, subgraph_id);
                            }
                        }
                    }
                }
            };
        collect_map_id(model, -1);
        ov::hetero::debug::dump_subgraphs(model, debug_supported_ops, map_id);
    }

    // Get subgraphs sorted topologically and appropriate mapping information
    return subgraph_collector.run();
}

void ov::hetero::fix_submodel_with_paged_attention(std::shared_ptr<ov::Model>& model) {
    for (auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::PagedAttentionExtension>(op)) {
            std::vector<std::shared_ptr<ov::Node>> reshape_nodes;
            for (size_t i = 0; i < 3; i++) {
                auto input_node = op->get_input_node_shared_ptr(i);
                auto input_value = input_node->input_value(0);
                const auto& shape = input_value.get_partial_shape();
                if (shape.rank().is_dynamic() || shape[2].is_dynamic() || shape[3].is_dynamic()) {
                    continue;
                }
                std::vector<int64_t> new_shape_values = {
                    -1,
                    static_cast<int>(shape[2].get_length() * shape[3].get_length())};
                auto shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                          ov::Shape{new_shape_values.size()},
                                                                          new_shape_values.data());

                auto new_reshape = std::make_shared<ov::op::v1::Reshape>(input_value, shape_const, false);
                new_reshape->set_friendly_name(input_node->get_friendly_name());
                ov::replace_node(input_node, new_reshape);
            }
        }
    }
}

ov::hetero::SubgraphsMappingInfo ov::hetero::mask_model_subgraphs_by_ops(std::shared_ptr<ov::Model>& model,
                                                                         ov::SupportedOpsMap& supported_ops,
                                                                         const bool dump_dot_files,
                                                                         const std::string default_device) {
    const std::string subgraph_id_rt_info_name = "HETERO_SUBGRAPH_ID";
    const std::string input_id_rt_info_name = "HETERO_INPUT_ID";
    const std::string output_id_rt_info_name = "HETERO_OUTPUT_ID";
    const auto& name = model->get_friendly_name();

    SubgraphsVector ordered_subgraphs;
    SubgraphsMappingInfo mapping_info;
    std::tie(ordered_subgraphs, mapping_info) =
        get_model_subgraphs(model, supported_ops, false, dump_dot_files, default_device);

    SubmodelsVector submodels{ordered_subgraphs.size()};
    for (size_t i = 0; i < ordered_subgraphs.size(); i++) {
        const auto& subgraph = ordered_subgraphs.at(i);
        auto submodel_name = name + '_' + std::to_string(i);
        submodels[i] =
            std::make_shared<ov::Model>(subgraph._results, subgraph._sinks, subgraph._parameters, submodel_name);
        const auto& submodel = submodels[i];
        // Check whether model is subgraph already
        bool is_subgraph = ov::op::util::has_op_with_type<ov::hetero::op::DeviceSubgraph>(submodel);
        if (is_subgraph) {
            for (auto& op : submodel->get_ordered_ops()) {
                if (!ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op) && !ov::op::util::is_parameter(op) &&
                    !ov::op::util::is_output(op) && !ov::op::util::is_sink(op)) {
                    is_subgraph = false;
                    break;
                }
            }
        }
        if (subgraph._affinity != default_device && !is_subgraph) {
            // Replace submodel by subgraph operation
            ParameterVector subgraph_parameters{submodel->inputs().size()};
            OutputVector args{submodel->inputs().size()};
            for (size_t j = 0; j < submodel->inputs().size(); j++) {
                const auto& input = submodel->input(j);
                subgraph_parameters[j] =
                    std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input.get_partial_shape());
                supported_ops[subgraph_parameters[j]->get_friendly_name()] = subgraph._affinity;
                args[j] = subgraph_parameters[j]->output(0);
            }
            auto subgraph_op = std::make_shared<ov::hetero::op::DeviceSubgraph>(args, submodel, subgraph._affinity);
            supported_ops[subgraph_op->get_friendly_name()] = subgraph._affinity;
            ResultVector subgraph_results{subgraph_op->outputs().size()};
            for (size_t j = 0; j < subgraph_op->outputs().size(); j++) {
                const auto& output = subgraph_op->output(j);
                subgraph_results[j] = std::make_shared<ov::op::v0::Result>(output);
                supported_ops[subgraph_results[j]->get_friendly_name()] = subgraph._affinity;
            }
            submodels[i] = std::make_shared<ov::Model>(subgraph_results, subgraph_parameters);
        }
        // Store original subgraph id
        for (auto& op : submodels[i]->get_ordered_ops()) {
            if (auto subgraph_op = ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op)) {
                auto& rt_info = op->get_rt_info();
                rt_info[subgraph_id_rt_info_name] = i;

                const auto& parameters = submodels[i]->get_parameters();
                OPENVINO_ASSERT(parameters.size() == op->inputs().size());
                for (auto& input : op->inputs()) {
                    const auto& source_output = input.get_source_output().get_node()->shared_from_this();
                    if (auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(source_output)) {
                        input.get_rt_info()[input_id_rt_info_name] = get_index(parameters, parameter);
                    }
                }

                const auto& results = submodels[i]->get_results();
                OPENVINO_ASSERT(results.size() == op->outputs().size());
                for (auto& output : op->outputs()) {
                    for (auto& input : output.get_target_inputs()) {
                        auto target_input = input.get_node()->shared_from_this();
                        if (auto result = ov::as_type_ptr<ov::op::v0::Result>(target_input)) {
                            output.get_rt_info()[output_id_rt_info_name] = get_index(results, result);
                        }
                    }
                }
            }
        }
    }

    merge_submodels(submodels, mapping_info._submodels_input_to_prev_output);

    model = submodels[0];
    fix_submodel_with_paged_attention(model);
    // Finally update mapping information according to the new operation order
    std::map<size_t, size_t> subgraph_id_map;
    std::map<size_t, std::map<size_t, size_t>> input_id_map;
    std::map<size_t, std::map<size_t, size_t>> output_id_map;
    size_t subgraph_op_id = 0;
    for (auto& op : model->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op)) {
            auto& rt_info = op->get_rt_info();
            subgraph_id_map[rt_info[subgraph_id_rt_info_name].as<size_t>()] = subgraph_op_id;
            rt_info.erase(subgraph_id_rt_info_name);
            for (size_t j = 0; j < op->inputs().size(); j++) {
                auto& input_rt_info = op->input(j).get_rt_info();
                input_id_map[subgraph_op_id][input_rt_info[input_id_rt_info_name].as<size_t>()] = j;
                input_rt_info.erase(input_id_rt_info_name);
            }
            for (size_t j = 0; j < op->outputs().size(); j++) {
                auto& output_rt_info = op->output(j).get_rt_info();
                output_id_map[subgraph_op_id][output_rt_info[output_id_rt_info_name].as<size_t>()] = j;
                output_rt_info.erase(output_id_rt_info_name);
            }
            subgraph_op_id++;
        }
    }
    SubgraphsMappingInfo new_mapping_info;
    if (ordered_subgraphs.size() == subgraph_op_id) {
        // Only if all subgraphs were replaced by subgraph operations
        // we can provide appropriate mapping information
        // otherwise this information is unavailable
        auto get_new_subgraph_index = [&subgraph_id_map](const size_t old_subgraph_index) {
            OPENVINO_ASSERT(subgraph_id_map.count(old_subgraph_index));
            return subgraph_id_map.at(old_subgraph_index);
        };
        auto get_new_input_index = [&input_id_map](const size_t subgraph_index, const size_t old_input_index) {
            OPENVINO_ASSERT(input_id_map.at(subgraph_index).count(old_input_index));
            return input_id_map.at(subgraph_index).at(old_input_index);
        };
        auto get_new_output_index = [&output_id_map](const size_t subgraph_index, const size_t old_output_index) {
            OPENVINO_ASSERT(output_id_map.at(subgraph_index).count(old_output_index));
            return output_id_map.at(subgraph_index).at(old_output_index);
        };
        for (auto& item : mapping_info._inputs_to_submodels_inputs) {
            const auto& subgraph_index = get_new_subgraph_index(item.first);
            const auto& input_index = get_new_input_index(subgraph_index, item.second);
            new_mapping_info._inputs_to_submodels_inputs.push_back({subgraph_index, input_index});
        }
        for (auto& item : mapping_info._outputs_to_submodels_outputs) {
            const auto& subgraph_index = get_new_subgraph_index(item.first);
            const auto& output_index = get_new_output_index(subgraph_index, item.second);
            new_mapping_info._outputs_to_submodels_outputs.push_back({subgraph_index, output_index});
        }
        for (auto& item : mapping_info._submodels_input_to_prev_output) {
            const auto& subgraph_in_index = get_new_subgraph_index(item.first.first);
            const auto& input_index = get_new_input_index(subgraph_in_index, item.first.second);

            const auto& subgraph_out_index = get_new_subgraph_index(item.second.first);
            const auto& output_index = get_new_output_index(subgraph_out_index, item.second.second);

            new_mapping_info._submodels_input_to_prev_output[{subgraph_in_index, input_index}] = {subgraph_out_index,
                                                                                                  output_index};
        }
    }
    return new_mapping_info;
}
