// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pass_manager.h"
#include "test_utils.h"

using namespace cldnn;

namespace {

using edge = std::pair<size_t, size_t>;

struct dependency_graph {
    engine& engine_ref = tests::get_test_engine();
    program prog{engine_ref, tests::get_test_default_config(engine_ref)};
    std::vector<program_node*> nodes;

    dependency_graph(size_t count,
                     const std::vector<edge>& edges,
                     std::vector<size_t> order = {},
                     const std::vector<size_t>& constants = {},
                     const std::vector<size_t>& skippable = {}) {
        const layout scalar({1, 1, 1, 1}, data_types::f32, format::bfyx);
        for (size_t i = 0; i < count; ++i) {
            const auto id = "node" + std::to_string(i);
            std::vector<input_info> inputs;
            for (const auto& e : edges) {
                if (e.second == i)
                    inputs.emplace_back("node" + std::to_string(e.first));
            }
            std::shared_ptr<primitive> prim;
            if (std::find(constants.begin(), constants.end(), i) != constants.end()) {
                prim = std::make_shared<data>(id, engine_ref.allocate_memory(scalar));
            } else if (inputs.empty()) {
                prim = std::make_shared<input_layout>(id, scalar);
            } else {
                prim = std::make_shared<concatenation>(id, inputs, 1);
            }
            auto* node = &prog.get_or_create(prim);
            node->set_unique_id(i + 10);
            if (std::find(skippable.begin(), skippable.end(), i) != skippable.end()) {
                node->can_be_optimized(true);
                node->set_runtime_skippable(true);
            }
            nodes.push_back(node);
        }
        for (const auto& e : edges)
            prog.add_connection(*nodes[e.first], *nodes[e.second], 0);

        // Load an explicit processing order, including deliberately invalid orders for fallback tests.
        if (order.empty()) {
            order.resize(count);
            std::iota(order.begin(), order.end(), size_t{0});
        }
        std::stringstream storage;
        BinaryOutputBuffer output(storage);
        output << order.size();
        for (auto it = order.rbegin(); it != order.rend(); ++it)
            output << nodes[*it]->id();
        BinaryInputBuffer input(storage, engine_ref);
        prog.get_processing_order().load(input, prog);
    }

    std::vector<std::vector<uint32_t>> restrictions() const {
        std::vector<std::vector<uint32_t>> result;
        for (auto* node : nodes)
            result.push_back(node->get_memory_dependencies());
        return result;
    }
};

// Independent traversal oracle; restriction expansion uses the unchanged shared pass helper.
void add_reference_restrictions(program& prog) {
    std::vector<program_node*> nodes;
    std::unordered_map<program_node*, size_t> index;
    for (auto* node : prog.get_processing_order()) {
        if (!node->is_type<data>()) {
            index.emplace(node, nodes.size());
            nodes.push_back(node);
        }
    }
    const auto count = nodes.size();
    std::vector<std::vector<bool>> reachable(count, std::vector<bool>(count, false));
    for (size_t source = 0; source < count; ++source) {
        std::vector<program_node*> pending(nodes[source]->get_users().begin(), nodes[source]->get_users().end());
        while (!pending.empty()) {
            auto* node = pending.back();
            pending.pop_back();
            const auto target = index.at(node);
            if (reachable[source][target])
                continue;
            reachable[source][target] = true;
            pending.insert(pending.end(), node->get_users().begin(), node->get_users().end());
        }
    }

    basic_memory_dependencies pass;
    const auto restrict_pair = [&](program_node* a, program_node* b) {
        pass.add_memory_dependency(a, b);
        pass.add_memory_dependency(b, a);
    };
    for (size_t a = 0; a < count; ++a) {
        const auto& dependencies = nodes[a]->get_dependencies();
        const auto nonconstant = std::count_if(dependencies.begin(), dependencies.end(), [](const auto& dep) {
            return !dep.first->is_constant();
        });
        if (nonconstant > 1) {
            std::vector<size_t> deps;
            for (const auto& dep : dependencies) {
                if (!dep.first->is_type<data>())
                    deps.push_back(index.at(dep.first));
            }
            std::sort(deps.begin(), deps.end());
            for (size_t i = 0; i < deps.size(); ++i) {
                for (size_t j = i + 1; j < deps.size(); ++j) {
                    if (reachable[deps[i]][deps[j]]) {
                        for (auto* user : nodes[deps[j]]->get_users())
                            restrict_pair(nodes[deps[i]], user);
                    }
                }
            }
        }

        // OOOQ handles each join before scanning pairs from the following non-data node.
        const auto source = a + 1;
        if (source == count)
            continue;
        for (size_t b = source + 1; b < count; ++b) {
            const bool other_branch = std::any_of(nodes[source]->get_users().begin(), nodes[source]->get_users().end(), [&](program_node* user) {
                const auto u = index.at(user);
                return u != b && !reachable[b][u] && !reachable[u][b];
            });
            if (!reachable[source][b] || other_branch)
                restrict_pair(nodes[source], nodes[b]);
        }
    }
}

void check_graph(size_t count,
                 const std::vector<edge>& edges,
                 const std::vector<size_t>& order = {},
                 const std::vector<size_t>& constants = {},
                 const std::vector<size_t>& skippable = {}) {
    dependency_graph actual(count, edges, order, constants, skippable);
    dependency_graph expected(count, edges, order, constants, skippable);
    add_reference_restrictions(expected.prog);
    oooq_memory_dependencies{}.run(actual.prog);
    EXPECT_EQ(actual.restrictions(), expected.restrictions());
}

TEST(oooq_memory_dependencies, empty_and_single_node) {
    check_graph(0, {});
    check_graph(1, {});
    check_graph(1, {}, {}, {0});
}

TEST(oooq_memory_dependencies, chain_across_bitmap_words) {
    std::vector<edge> edges;
    for (size_t i = 1; i < 130; ++i)
        edges.emplace_back(i - 1, i);
    check_graph(130, edges);
}

TEST(oooq_memory_dependencies, diamond_and_disconnected_components) {
    check_graph(9, {{0, 1}, {1, 2}, {1, 3}, {2, 4}, {3, 4}, {5, 6}, {6, 7}});
}

TEST(oooq_memory_dependencies, wide_layers) {
    std::vector<edge> edges;
    for (size_t layer = 0; layer < 3; ++layer) {
        for (size_t i = 0; i < 20; ++i) {
            for (size_t j = 0; j < 20; ++j)
                edges.emplace_back(layer * 20 + i, (layer + 1) * 20 + j);
        }
    }
    check_graph(80, edges);
}

TEST(oooq_memory_dependencies, connected_join_inputs_and_duplicate_edges) {
    check_graph(7, {{0, 1}, {1, 2}, {1, 2}, {1, 3}, {2, 3}, {2, 4}, {3, 5}, {4, 5}});
}

TEST(oooq_memory_dependencies, constants_and_runtime_skippable_concat) {
    check_graph(9, {{0, 2}, {1, 2}, {2, 3}, {3, 4}, {2, 5}, {4, 6}, {5, 6}, {6, 7}}, {}, {0}, {3, 4, 6});
}

TEST(oooq_memory_dependencies, computed_constants_remain_in_reachability) {
    const std::vector<edge> edges{{0, 2}, {2, 3}, {1, 4}, {3, 4}, {4, 5}, {1, 6}, {5, 7}, {6, 7}};
    dependency_graph actual(8, edges, {}, {0});
    dependency_graph expected(8, edges, {}, {0});
    for (auto* graph : {&actual, &expected}) {
        graph->prog.mark_if_constant(*graph->nodes[2]);
        graph->prog.mark_if_constant(*graph->nodes[3]);
        ASSERT_TRUE(graph->nodes[3]->is_constant());
    }
    add_reference_restrictions(expected.prog);
    oooq_memory_dependencies{}.run(actual.prog);
    EXPECT_EQ(actual.restrictions(), expected.restrictions());
}

TEST(oooq_memory_dependencies, processing_order_differs_from_node_ids) {
    check_graph(7, {{5, 3}, {3, 1}, {3, 6}, {1, 0}, {6, 0}, {4, 2}}, {5, 4, 3, 2, 6, 1, 0});
}

TEST(oooq_memory_dependencies, random_dags_match_traversal_oracle) {
    std::mt19937 rng(2026);
    for (size_t count : {2, 7, 31, 64, 65, 129}) {
        for (size_t sample = 0; sample < 8; ++sample) {
            SCOPED_TRACE(::testing::Message() << "nodes=" << count << " sample=" << sample);
            std::vector<edge> edges;
            for (size_t a = 0; a < count; ++a) {
                for (size_t b = a + 1; b < count; ++b) {
                    if (rng() % 11 == 0)
                        edges.emplace_back(a, b);
                }
            }
            check_graph(count, edges);
        }
    }
}

// Frozen legacy results: non-topological inputs must retain the existing fixed-point behavior.
TEST(oooq_memory_dependencies, backward_edge_falls_back_to_legacy_behavior) {
    dependency_graph graph(4, {{1, 0}, {0, 2}, {2, 3}});
    oooq_memory_dependencies{}.run(graph.prog);
    EXPECT_EQ(graph.restrictions(), (std::vector<std::vector<uint32_t>>{{}, {12, 13}, {11}, {11}}));
}

TEST(oooq_memory_dependencies, cycle_falls_back_to_legacy_behavior) {
    dependency_graph graph(4, {{0, 1}, {1, 2}, {2, 1}, {2, 3}});
    oooq_memory_dependencies{}.run(graph.prog);
    EXPECT_EQ(graph.restrictions(), (std::vector<std::vector<uint32_t>>{{11, 13}, {10}, {}, {10}}));
}

TEST(oooq_memory_dependencies, self_edge_falls_back_to_legacy_behavior) {
    dependency_graph graph(4, {{0, 1}, {1, 1}, {1, 2}, {2, 3}});
    oooq_memory_dependencies{}.run(graph.prog);
    EXPECT_EQ(graph.restrictions(), (std::vector<std::vector<uint32_t>>{{11, 12}, {10}, {10}, {}}));
}

}  // namespace
