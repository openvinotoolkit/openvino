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

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/runtime/memory_pool.hpp"
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
                     const std::vector<size_t>& skippable = {},
                     const std::vector<size_t>& shape_ofs = {}) {
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
            } else if (std::find(shape_ofs.begin(), shape_ofs.end(), i) != shape_ofs.end()) {
                OPENVINO_ASSERT(inputs.size() == 1, "shape_of test node requires exactly one input");
                prim = std::make_shared<shape_of>(id, inputs.front(), data_types::i64);
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
            // Edges into shape_of do not contribute to memory-lifetime reachability:
            // shape_of reads only the input layout and does not enforce runtime ordering.
            if (node->is_type<shape_of>())
                continue;
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
                 const std::vector<size_t>& skippable = {},
                 const std::vector<size_t>& shape_ofs = {}) {
    dependency_graph actual(count, edges, order, constants, skippable, shape_ofs);
    dependency_graph expected(count, edges, order, constants, skippable, shape_ofs);
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

// CVS-195973 reproducer: convert -> stft -> shape_of -> subtract (ONNX STFT with runtime frame_step).
// The only path from convert to subtract goes through shape_of, which reads no input data and
// does not enforce runtime ordering, so their buffers must not be shared.
// A leading dummy node keeps 'convert' away from the pair loop's first-node exemption.
TEST(oooq_memory_dependencies, shape_of_edge_does_not_close_buffer_lifetime) {
    dependency_graph graph(5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}}, {}, {}, {}, {3});
    oooq_memory_dependencies{}.run(graph.prog);
    // Without the fix every restriction list is empty: convert(11) and subtract(14) may share a buffer.
    EXPECT_EQ(graph.restrictions(), (std::vector<std::vector<uint32_t>>{{}, {13, 14}, {13, 14}, {11, 12}, {11, 12}}));
}

// A node feeding both a data consumer and a shape_of keeps restrictions against downstream nodes.
TEST(oooq_memory_dependencies, shape_of_sibling_branch_matches_traversal_oracle) {
    check_graph(5, {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}}, {}, {}, {}, {2});
}

TEST(oooq_memory_dependencies, random_dags_with_shape_of_match_traversal_oracle) {
    std::mt19937 rng(4242);
    for (size_t count : {3, 8, 33, 66}) {
        for (size_t sample = 0; sample < 6; ++sample) {
            SCOPED_TRACE(::testing::Message() << "nodes=" << count << " sample=" << sample);
            std::vector<edge> edges;
            for (size_t a = 0; a < count; ++a) {
                for (size_t b = a + 1; b < count; ++b) {
                    if (rng() % 9 == 0)
                        edges.emplace_back(a, b);
                }
            }
            // Turn a few nodes into shape_of, keeping exactly one input edge for each.
            std::vector<size_t> shape_ofs;
            for (size_t s = 1; s < count && shape_ofs.size() < 3; ++s) {
                std::vector<edge> incoming;
                std::vector<edge> rest;
                for (const auto& e : edges)
                    (e.second == s ? incoming : rest).push_back(e);
                if (incoming.empty())
                    continue;
                edges = rest;
                edges.emplace_back(incoming[rng() % incoming.size()].first, s);
                shape_ofs.push_back(s);
            }
            check_graph(count, edges, {}, {}, {}, shape_ofs);
        }
    }
}

// Pool-level counterpart: pins the pool's adherence to restrictions. Given two same-sized
// allocations, the memory pool aliases the first buffer for the second node unless the
// second node's restriction set contains the first node's unique id. The pass tests above
// pin that the fix generates such a restriction for edges into shape_of; this test uses a
// manually built restriction set, so the two together verify restriction generation and pool
// adherence separately (the node -> pool plumbing, node.get_memory_dependencies() feeding
// the allocation request, is verified at code level in primitive_inst.cpp).
TEST(oooq_memory_dependencies, pool_reuses_buffer_unless_restricted) {
    auto& engine = tests::get_test_engine();
    auto config = tests::get_test_default_config(engine);
    memory_pool pool(engine, config);

    const layout l({1, 1, 1, 4}, data_types::i32, format::bfyx);
    const auto type = allocation_type::usm_device;

    memory_restricter<uint32_t> none;
    auto a = pool.get_memory(l, "a", 11, 0, none, type);
    ASSERT_NE(a, nullptr);

    // Without a restriction on a's unique id, the pool hands b the very same buffer.
    memory_restricter<uint32_t> unrestricted;
    auto b = pool.get_memory(l, "b", 12, 0, unrestricted, type);
    ASSERT_NE(b, nullptr);
    ASSERT_EQ(a->get_internal_params().mem, b->get_internal_params().mem);

    // With the (a, b) restriction that the fixed pass generates, the pool must allocate
    // a distinct buffer.
    memory_restricter<uint32_t> restricted;
    restricted.insert(11);
    auto c = pool.get_memory(l, "c", 13, 0, restricted, type);
    ASSERT_NE(c, nullptr);
    EXPECT_NE(a->get_internal_params().mem, c->get_internal_params().mem);
}

// Network-level counterpart: pins the implementation selection that makes buffer aliasing
// across a ShapeOf edge unsafe: shape_of runs as a CPU impl that reads only the input layout
// and does not wait for its input producer to complete (its marked subgraph nodes also run
// on CPU; other CPU impls may wait on input events, but CPU eltwise skips that wait for an
// out-of-order queue with CPU-only dependencies). Guards that a real network execution keeps
// the two data-path buffers distinct end to end.
//
// Scope caveats:
// - The graph mirrors the STFT shape-subgraph *structure* (OCL eltwise data nodes feeding a
//   CPU shape_of chain, mirroring Convert_3834 -> ShapeOf_25 -> ... -> Subtract_27 of
//   CVS-195973); it does not contain literal STFT decomposition ops. The ONNX functional
//   test exercises the real STFT model end to end (output correctness) but does not itself
//   assert which implementations are selected.
// - This is a post-fix invariant guard, not a pre-fix reproducer: in this static synthetic
//   graph the (a, b) pair is additionally covered by skipped_branch_memory_dependencies and
//   buffer lifetime rules, so the pre-fix aliasing itself is reproduced by the dynamic ONNX
//   model, not here. The restriction -> pool causality is pinned by
//   pool_reuses_buffer_unless_restricted above. The restriction-set assertion below only
//   guards the node -> pool plumbing (the node's memory_dependencies, the pool's input,
//   carries the (b, a) no-share id); in this graph that id can also come from other passes,
//   so it is not fix-specific.
// - 'out' consumes the data of both 'a' and 'b', so their GPU kernels cannot be runtime-
//   skipped and their outputs stay pool-allocated; the value checks below fail if either
//   primitive is skipped or its buffer is corrupted.
TEST(oooq_memory_dependencies, shape_of_chain_runtime_pool_keeps_buffers_distinct) {
    auto& engine = tests::get_test_engine();

    // tensor(b, f, x, y) maps to PartialShape [b,f,y,x]: input/c1 are [4,1,4,1] and
    // input2/c3 are [4,1,1,4], so 'a' and 'b' both end up 16-element i32 tensors (64B)
    // and the pool could only satisfy one of them by reusing the other's buffer.
    // shape_of outputs a rank-1 {rank} tensor, so kc is [4,1,1,1] to keep broadcasts valid.
    std::vector<int32_t> iota(16);
    std::iota(iota.begin(), iota.end(), 1);

    auto input = engine.allocate_memory({data_types::i32, format::bfyx, tensor{4, 1, 1, 4}});
    tests::set_values(input, iota);
    auto input2 = engine.allocate_memory({data_types::i32, format::bfyx, tensor{4, 1, 4, 1}});
    tests::set_values(input2, iota);

    auto c1 = engine.allocate_memory({data_types::i32, format::bfyx, tensor{4, 1, 1, 4}});
    tests::set_values(c1, std::vector<int32_t>(16, 1));
    auto kc = engine.allocate_memory({data_types::i32, format::bfyx, tensor{4, 1, 1, 1}});
    tests::set_values(kc, std::vector<int32_t>(4, 1));
    auto c3 = engine.allocate_memory({data_types::i32, format::bfyx, tensor{4, 1, 4, 1}});
    tests::set_values(c3, std::vector<int32_t>(16, 1));

    topology topo;
    topo.add(input_layout("input", input->get_layout()));
    topo.add(input_layout("input2", input2->get_layout()));
    topo.add(data("c1", c1));
    topo.add(data("kc", kc));
    topo.add(data("c3", c3));
    // Data-path producer feeding a shape_of node (mirrors the STFT Convert_3834); its only
    // *shape* consumer is 'so', the other user ('out') keeps it executing and pool-allocated.
    topo.add(eltwise("a", {input_info("input"), input_info("c1")}, eltwise_mode::sub));
    topo.add(shape_of("so", input_info("a"), data_types::i32));
    topo.add(eltwise("m1", {input_info("so"), input_info("kc")}, eltwise_mode::prod));
    // The second, unmarked data input keeps 'b' outside the shape_of subgraph, so its output
    // stays pool-allocated like 'a' (mirrors the STFT Subtract).
    topo.add(eltwise("a2", {input_info("input2"), input_info("c3")}, eltwise_mode::sub));
    topo.add(eltwise("b", {input_info("m1"), input_info("a2")}, eltwise_mode::sub));
    topo.add(shape_of("so2", input_info("b"), data_types::i32));
    // 'out' consumes the actual data of both 'a' and 'b' (two users each: they are neither
    // runtime-skippable nor fusable) so the value checks below prove real GPU execution.
    topo.add(eltwise("out", {input_info("a"), input_info("b")}, eltwise_mode::sum));

    auto config = tests::get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::enable_memory_pool(true));
    network net(engine, topo, config);
    net.set_input_data("input", input);
    net.set_input_data("input2", input2);
    auto outputs = net.execute();

    // Implementation selection: shape_of nodes and the marked subgraph node run on CPU.
    // The CPU shape_of impl reads only the input layout and does not wait for 'a' to
    // complete (other CPU impls may wait on input events; CPU eltwise skips that wait only
    // for an out-of-order queue with CPU-only dependencies).
    for (const auto* id : {"so", "so2", "m1"}) {
        auto inst = net.get_primitive(id);
        ASSERT_NE(inst, nullptr) << id;
        EXPECT_TRUE(inst->get_impl()->is_cpu()) << id;
    }
    // 'a' and 'b' remain data-path (non-CPU) nodes, and the address check below is only
    // meaningful if their outputs really come from the reusable pool: pin the pool being
    // enabled and the nodes being pool-eligible (not optimized away, not an output,
    // shareable). Otherwise buffers would be distinct regardless of the pass.
    EXPECT_TRUE(net.get_config().get_enable_memory_pool());
#ifdef ENABLE_DEBUG_CAPS
    EXPECT_FALSE(net.get_config().get_disable_memory_reuse());
#endif
    for (const auto* id : {"a", "b"}) {
        auto inst = net.get_primitive(id);
        ASSERT_NE(inst, nullptr) << id;
        EXPECT_FALSE(inst->get_impl()->is_cpu()) << id;
        EXPECT_FALSE(inst->get_node().can_be_optimized()) << id;
        EXPECT_FALSE(inst->get_node().is_output()) << id;
        EXPECT_FALSE(inst->get_node().is_in_shape_of_subgraph()) << id;
        EXPECT_TRUE(inst->get_node().can_share_buffer()) << id;
    }

    // Buffer identity: the pool must not alias the output buffers of 'a' and 'b', whose
    // only graph connection goes through the shape_of subgraph. The two buffers are pool
    // aliasing candidates (same allocation type and size), and 'b's allocation request
    // carries the no-share restriction on 'a' - the exact input the pool consults.
    const auto& b_node = net.get_primitive("b")->get_node();
    const auto a_uid = net.get_primitive("a")->get_node().get_unique_id();
    const auto& b_deps = b_node.get_memory_dependencies();
    EXPECT_NE(std::find(b_deps.begin(), b_deps.end(), a_uid), b_deps.end()) << "the (b, a) no-share restriction must survive on the network node";
    auto a_mem = net.get_output_memory("a");
    auto b_mem = net.get_output_memory("b");
    ASSERT_NE(a_mem, nullptr);
    ASSERT_NE(b_mem, nullptr);
    EXPECT_EQ(a_mem->get_allocation_type(), b_mem->get_allocation_type());
    EXPECT_EQ(a_mem->get_layout().bytes_count(), b_mem->get_layout().bytes_count());
    EXPECT_NE(a_mem->get_internal_params().mem, b_mem->get_internal_params().mem);

    // Execution correctness: 'out' = a + b consumes the real data of both data-path
    // primitives, so this fails if either kernel is skipped or its buffer is aliased.
    // a[bi][y] = 4*bi + y; b[bi][x] = m1v[x] - (4*bi + x) with m1v = shape_of(a) = {4,1,4,1};
    // hence out[bi][y][x] = y + m1v[x] - x, independent of bi.
    const std::vector<int32_t> m1v = {4, 1, 4, 1};
    auto out_mem = outputs.at("out").get_memory();
    cldnn::mem_lock<int32_t, mem_lock_type::read> out_ptr(out_mem, tests::get_test_stream());
    size_t idx = 0;
    for (int bi = 0; bi < 4; ++bi)
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x, ++idx)
                ASSERT_TRUE(tests::are_equal(y + m1v[x] - x, out_ptr[idx])) << "out[" << bi << "][" << y << "][" << x << "]";

    // so2 = shape_of(b), b has shape {4,1,1,4}.
    auto output = outputs.at("so2").get_memory();
    cldnn::mem_lock<int32_t, mem_lock_type::read> so2_ptr(output, tests::get_test_stream());
    std::vector<int32_t> expected = {4, 1, 1, 4};
    for (size_t i = 0; i < expected.size(); ++i)
        ASSERT_TRUE(tests::are_equal(expected[i], so2_ptr[i]));
}

}  // namespace
