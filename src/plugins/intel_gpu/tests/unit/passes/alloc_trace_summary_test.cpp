// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// B0/D0 minimal subset: GPU_DEBUG allocation trace and off-pool classification (dump_alloc_trace option).
// Every statistic is checked on a synthetic graph with a known construction. The category priority
//   shape_of > kv_variable > network_output > [internal_buffer] > inplace_concat_group > sum_root > multi_output_port > constant > other
// is fixed by the orchestrator and pinned by the classifier tests below.

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "arg_max_min_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/arg_max_min.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/primitives/softmax.hpp"
#include "intel_gpu/runtime/memory_pool.hpp"
#include "primitive_inst.h"
#include "program_helpers.h"
#include "program_node.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

#ifdef GPU_DEBUG_CONFIG

namespace {

ExecutionConfig trace_cfg(engine& eng, int trace_level = 2, const std::vector<std::string>& outputs = {}) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    if (trace_level > 0) {
        config.set_user_property({ov::intel_gpu::dump_alloc_trace(trace_level)});
    }
    if (!outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(outputs));
    }
    return config;
}

const alloc_trace_record* find_record(const std::vector<alloc_trace_record>& recs, const primitive_id& id, size_t port = 0, bool internal = false) {
    for (const auto& r : recs) {
        if (r.node_id == id && r.port == port && r.is_internal_buffer == internal) {
            return &r;
        }
    }
    return nullptr;
}

std::string dump_records(const std::vector<alloc_trace_record>& recs) {
    std::string out;
    for (const auto& r : recs) {
        out += "  " + r.node_id + " port=" + std::to_string(r.port) + " bytes=" + std::to_string(r.bytes) + " pooled=" + std::to_string(r.pooled) +
               " cat=" + to_string(r.category) + " reason=" + r.reason + "\n";
    }
    return out;
}

uint64_t bytes_of(network& net, const primitive_id& id, size_t port = 0) {
    return net.get_primitive(id)->get_output_layout(port).bytes_count();
}

// arg_max_min with two outputs (values, indices); output 0 of the node is left unused.
topology multi_output_topology(engine& eng) {
    auto input_mem = eng.allocate_memory({data_types::f32, format::bfyx, {2, 4, 2, 2}});
    auto top_k_input = eng.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}});
    topology t;
    t.add(input_layout("input", input_mem->get_layout()));
    t.add(cldnn::data("const", {top_k_input}));
    auto arg = arg_max_min("arg_max", {input_info("input"), input_info("const")}, ov::op::TopKMode::MAX, 2, 0, ov::op::TopKSortType::SORT_VALUES, false, false,
                           data_types::f32, 2);
    arg.output_paddings = {padding(), padding()};
    arg.output_data_types = {optional_data_type{data_types::f32}, optional_data_type{data_types::f32}};
    t.add(arg);
    t.add(activation("R", input_info("arg_max", 1), activation_func::abs));
    t.add(activation("O", input_info("R"), activation_func::abs));
    return t;
}

}  // namespace

// ----------------------------------------------------------------------------------------------------------------------------------
// Network outputs and constants.
// ----------------------------------------------------------------------------------------------------------------------------------
TEST(alloc_trace_summary, network_output_and_constant) {
    auto& eng = get_test_engine();
    layout l{ov::PartialShape{1, 3, 4, 4}, data_types::f32, format::bfyx};
    topology t;
    t.add(input_layout("A", l));
    t.add(activation("p", input_info("A"), activation_func::relu));
    t.add(activation("out", input_info("p"), activation_func::abs));
    auto prog = program::build_program(eng, t, trace_cfg(eng), false, false);
    network net(prog, 0);
    const auto s = net.get_alloc_trace_summary();
    const auto recs = net.get_alloc_trace_records();
    std::cout << dump_records(recs);

    const auto out_bytes = bytes_of(net, "out");
    EXPECT_EQ(s.at(alloc_category::network_output).count, 1u);
    EXPECT_EQ(s.at(alloc_category::network_output).bytes, out_bytes);
    EXPECT_EQ(s.network_output_bytes, out_bytes);
    const auto* rec = find_record(recs, "out");
    ASSERT_NE(rec, nullptr);
    EXPECT_EQ(rec->category, alloc_category::network_output);
    EXPECT_FALSE(rec->pooled);

    // input_layout buffers are bound by set_input_data and are not allocated by allocate_output (constant category: see classifier test)
    EXPECT_EQ(find_record(recs, "A"), nullptr);
    EXPECT_EQ(s.at(alloc_category::constant).count, 0u);

    // the intermediate comes from the pool
    EXPECT_EQ(s.at(alloc_category::pooled).count, 1u);
    EXPECT_EQ(s.at(alloc_category::pooled).bytes, bytes_of(net, "p"));
    EXPECT_EQ(s.total_count, 2u);
    EXPECT_EQ(s.total_bytes, out_bytes + bytes_of(net, "p"));
    EXPECT_EQ(s.off_pool_bytes(), out_bytes);
    // network output share of the allocated bytes: out / (out + p) = 1/2 for equal sizes
    EXPECT_EQ(s.network_output_bytes * 2, s.total_bytes);
    EXPECT_EQ(s.multi_output_node_uids.size(), 0u);
}

// ----------------------------------------------------------------------------------------------------------------------------------
// In-place concat: the holder (cbo concat) and its can_share_buffer(false) predecessors are one group.
// ----------------------------------------------------------------------------------------------------------------------------------
TEST(alloc_trace_summary, inplace_concat_group_counts_holder_and_predecessors) {
    auto& eng = get_test_engine();
    layout la{ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx};
    layout lb{ov::PartialShape{1, 3, 2, 2}, data_types::f32, format::bfyx};
    topology t;
    t.add(input_layout("A", la));
    t.add(input_layout("B", lb));
    t.add(activation("pa", input_info("A"), activation_func::relu));
    t.add(activation("pb", input_info("B"), activation_func::relu));
    t.add(concatenation("c", {input_info("pa"), input_info("pb")}, 1));
    t.add(activation("out", input_info("c"), activation_func::abs));
    auto prog = program::build_program(eng, t, trace_cfg(eng), false, false);
    ASSERT_TRUE(prog->get_node("c").can_be_optimized()) << "in-place concat expected";
    ASSERT_FALSE(prog->get_node("pa").can_share_buffer());
    ASSERT_FALSE(prog->get_node("pb").can_share_buffer());
    network net(prog, 0);
    const auto s = net.get_alloc_trace_summary();
    const auto recs = net.get_alloc_trace_records();
    std::cout << dump_records(recs);

    // Only the holder allocates: the predecessors are can_share_buffer(false) zero-allocation views of the holder buffer, so they produce no
    // allocation event; the holder bytes are the whole group.
    EXPECT_EQ(s.at(alloc_category::inplace_concat_group).count, 1u);
    EXPECT_EQ(s.at(alloc_category::inplace_concat_group).bytes, bytes_of(net, "c"));
    const auto* r = find_record(recs, "c");
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->category, alloc_category::inplace_concat_group);
    EXPECT_FALSE(r->pooled);
    EXPECT_EQ(find_record(recs, "pa"), nullptr);
    EXPECT_EQ(find_record(recs, "pb"), nullptr);
    // padded predecessor layouts span the whole holder buffer
    EXPECT_EQ(bytes_of(net, "pa"), bytes_of(net, "c"));
    EXPECT_EQ(bytes_of(net, "pb"), bytes_of(net, "c"));
    // the predecessors would be classified as the same group if they allocated (priority 4 beats 5)
    EXPECT_EQ(classify_off_pool_allocation(prog->get_node("pa"), false, 0, false), alloc_category::inplace_concat_group);
    EXPECT_EQ(classify_off_pool_allocation(prog->get_node("pb"), false, 0, false), alloc_category::inplace_concat_group);
    // priority 4 beats priority 5: the predecessors are can_share_buffer(false) yet are not counted as sum roots
    EXPECT_EQ(s.at(alloc_category::sum_root).count, 0u);
}

// ----------------------------------------------------------------------------------------------------------------------------------
// sum_root: structural variant (no oneDNN needed): a node with can_share_buffer(false) that is not constant-like.
// ----------------------------------------------------------------------------------------------------------------------------------
TEST(alloc_trace_summary, sum_root_structural) {
    auto& eng = get_test_engine();
    layout l{ov::PartialShape{1, 3, 4, 4}, data_types::f32, format::bfyx};
    topology t;
    t.add(input_layout("A", l));
    t.add(activation("root", input_info("A"), activation_func::relu));
    t.add(activation("out", input_info("root"), activation_func::abs));
    auto prog = program::build_program(eng, t, trace_cfg(eng), false, false);
    prog->get_node("root").can_share_buffer(false);
    network net(prog, 0);
    const auto s = net.get_alloc_trace_summary();
    const auto recs = net.get_alloc_trace_records();
    std::cout << dump_records(recs);
    EXPECT_EQ(s.at(alloc_category::sum_root).count, 1u);
    EXPECT_EQ(s.at(alloc_category::sum_root).bytes, bytes_of(net, "root"));
    const auto* r = find_record(recs, "root");
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->category, alloc_category::sum_root);
    EXPECT_STREQ(r->reason, "bypass_pool");
    EXPECT_EQ(s.at(alloc_category::pooled).count, 0u);
}

// sum_root: real oneDNN residual chain  x_proj -> fc_1 -> ... -> p (fused sum with x_proj) -> softmax
#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(alloc_trace_summary, sum_root_onednn_chain) {
    auto& eng = get_test_engine();
    if (!eng.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN sum post-op requires supports_immad";
    }
    auto make_w = [&](int64_t ofm, int64_t ifm) {
        auto m = eng.allocate_memory(layout{ov::PartialShape{ofm, ifm}, data_types::f32, format::bfyx});
        mem_lock<float> l(m, get_test_stream());
        for (size_t i = 0; i < l.size(); i++) {
            l[i] = 0.01f;
        }
        return m;
    };
    topology t;
    t.add(input_layout("in", layout{ov::PartialShape{32, 64}, data_types::f32, format::bfyx}));
    for (const char* id : {"x_proj", "fc_1", "p"}) {
        t.add(data(std::string(id) + "_w", make_w(64, 64)));
    }
    t.add(fully_connected("x_proj", input_info("in"), "x_proj_w"));
    t.add(fully_connected("fc_1", input_info("x_proj"), "fc_1_w"));
    t.add(fully_connected("p", input_info("fc_1"), "p_w"));
    t.add(eltwise("res", input_info("p"), input_info("x_proj"), eltwise_mode::sum));
    t.add(softmax("out", input_info("res"), 1));

    auto prog = program::build_program(eng, t, trace_cfg(eng), false, false);
    auto& p = prog->get_node("p");
    auto& x = prog->get_node("x_proj");
    if (p.get_preferred_impl_type() != impl_types::onednn || x.can_share_buffer() || p.can_share_buffer()) {
        GTEST_SKIP() << "oneDNN in-place sum was not selected for this graph: p.can_share=" << p.can_share_buffer() << " x.can_share=" << x.can_share_buffer();
    }
    network net(prog, 0);
    const auto s = net.get_alloc_trace_summary();
    const auto recs = net.get_alloc_trace_records();
    std::cout << dump_records(recs);
    // The chain root allocates; the sum node p writes into the root buffer (reused eltwise dependency memory) and does not allocate.
    EXPECT_EQ(s.at(alloc_category::sum_root).count, 1u);
    EXPECT_EQ(s.at(alloc_category::sum_root).bytes, bytes_of(net, "x_proj"));
    const auto* r = find_record(recs, "x_proj");
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->category, alloc_category::sum_root);
    EXPECT_EQ(find_record(recs, "p"), nullptr);
    EXPECT_EQ(classify_off_pool_allocation(p, false, 0, false), alloc_category::sum_root);
}
#else
TEST(alloc_trace_summary, sum_root_onednn_chain) {
    GTEST_SKIP() << "built without oneDNN";
}
#endif

// ----------------------------------------------------------------------------------------------------------------------------------
// Multi-output node: idx>0 statistics, pooled ports and network-output ports.
// ----------------------------------------------------------------------------------------------------------------------------------
TEST(alloc_trace_summary, multi_output_ports_pooled) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, multi_output_topology(eng), trace_cfg(eng), false, false);
    ASSERT_EQ(prog->get_node("arg_max").get_outputs_count(), 2u);
    network net(prog, 0);
    const auto s = net.get_alloc_trace_summary();
    const auto recs = net.get_alloc_trace_records();
    std::cout << dump_records(recs);
    EXPECT_EQ(s.multi_output_node_uids.size(), 1u);
    EXPECT_EQ(s.non_first_port_count, 1u);
    EXPECT_EQ(s.non_first_port_bytes, bytes_of(net, "arg_max", 1));
    for (size_t port : {size_t(0), size_t(1)}) {
        const auto* r = find_record(recs, "arg_max", port);
        ASSERT_NE(r, nullptr) << port;
        EXPECT_TRUE(r->pooled) << port;
        EXPECT_EQ(r->category, alloc_category::pooled) << port;
    }
    EXPECT_EQ(s.at(alloc_category::multi_output_port).count, 0u);
}

TEST(alloc_trace_summary, multi_output_ports_network_output) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, multi_output_topology(eng), trace_cfg(eng, 2, {"arg_max"}), false, false);
    ASSERT_TRUE(prog->get_node("arg_max").is_output());
    network net(prog, 0);
    const auto s = net.get_alloc_trace_summary();
    const auto recs = net.get_alloc_trace_records();
    std::cout << dump_records(recs);
    EXPECT_EQ(s.non_first_port_count, 1u);
    EXPECT_EQ(s.non_first_port_bytes, bytes_of(net, "arg_max", 1));
    for (size_t port : {size_t(0), size_t(1)}) {
        const auto* r = find_record(recs, "arg_max", port);
        ASSERT_NE(r, nullptr) << port;
        EXPECT_FALSE(r->pooled) << port;
        EXPECT_EQ(r->category, alloc_category::network_output) << port;
    }
    EXPECT_EQ(s.at(alloc_category::multi_output_port).count, 0u);
}

// ----------------------------------------------------------------------------------------------------------------------------------
// shape_of subgraph.
// ----------------------------------------------------------------------------------------------------------------------------------
TEST(alloc_trace_summary, shape_of_subgraph) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("input", layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}));
    t.add(shape_of("so", input_info("input"), data_types::i32));
    t.add(eltwise("se", {input_info("so"), input_info("so")}, eltwise_mode::sum));
    t.add(activation("out", input_info("se"), activation_func::abs));
    auto prog = program::build_program(eng, t, trace_cfg(eng), false, false);
    network net(prog, 0);
    auto in = eng.allocate_memory(layout{ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx});
    net.set_input_data("input", in);
    net.execute();
    const auto s = net.get_alloc_trace_summary();
    const auto recs = net.get_alloc_trace_records();
    std::cout << dump_records(recs);
    const auto* so = find_record(recs, "so");
    ASSERT_NE(so, nullptr) << "shape_of output allocation was not traced";
    EXPECT_EQ(so->category, alloc_category::shape_of);
    EXPECT_FALSE(so->pooled);
    EXPECT_GE(s.at(alloc_category::shape_of).count, 1u);
    // priority 1 beats 3: nodes of the shape_of subgraph are counted as shape_of even if they were network outputs
    EXPECT_EQ(s.at(alloc_category::shape_of).bytes,
              [&] {
                  uint64_t b = 0;
                  for (const auto& r : recs) {
                      b += r.category == alloc_category::shape_of ? r.bytes : 0;
                  }
                  return b;
              }());
}

// ----------------------------------------------------------------------------------------------------------------------------------
// Classifier priority (direct, on hand-flagged nodes).
// ----------------------------------------------------------------------------------------------------------------------------------
TEST(alloc_trace_summary, classifier_priority_and_multi_output_port) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, multi_output_topology(eng), trace_cfg(eng, 0), false, /*no_optimizations=*/true);
    auto& n = prog->get_node("arg_max");
    ASSERT_EQ(n.get_outputs_count(), 2u);
    ASSERT_TRUE(n.can_share_buffer());
    ASSERT_FALSE(n.is_output());

    // 6 multi_output_port / 8 other
    EXPECT_EQ(classify_off_pool_allocation(n, false, 0, false), alloc_category::other);
    EXPECT_EQ(classify_off_pool_allocation(n, false, 1, false), alloc_category::multi_output_port);
    // internal buffers rank below the first three categories only
    EXPECT_EQ(classify_off_pool_allocation(n, false, 0, true), alloc_category::internal_buffer);
    EXPECT_EQ(classify_off_pool_allocation(n, false, 1, true), alloc_category::internal_buffer);
    // 5 sum_root beats 6 multi_output_port
    n.can_share_buffer(false);
    EXPECT_EQ(classify_off_pool_allocation(n, false, 1, false), alloc_category::sum_root);
    EXPECT_EQ(classify_off_pool_allocation(n, false, 0, false), alloc_category::sum_root);
    // 3 network_output beats 5 and the internal buffer category
    n.set_output(true);
    EXPECT_EQ(classify_off_pool_allocation(n, false, 1, false), alloc_category::network_output);
    EXPECT_EQ(classify_off_pool_allocation(n, false, 0, true), alloc_category::network_output);
    n.set_output(false);

    // 7 constant-like nodes are never sum roots even though their can_share_buffer is false
    auto& in = prog->get_node("input");
    EXPECT_FALSE(in.can_share_buffer());
    EXPECT_EQ(classify_off_pool_allocation(in, false, 0, false), alloc_category::constant);
    in.set_output(true);
    EXPECT_EQ(classify_off_pool_allocation(in, false, 0, false), alloc_category::network_output);
    in.set_output(false);
}

TEST(alloc_trace_summary, classifier_kv_variable_beats_network_output_and_group) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("input", layout{ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx}));
    t.add(read_value("rv", std::vector<input_info>{}, "var0", {layout{ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx}}));
    t.add(eltwise("sum", {input_info("input"), input_info("rv")}, eltwise_mode::sum));
    auto prog = program::build_program(eng, t, trace_cfg(eng, 0), false, /*no_optimizations=*/true);
    auto& rv = prog->get_node("rv");
    EXPECT_EQ(classify_off_pool_allocation(rv, false, 0, false), alloc_category::kv_variable);
    rv.set_output(true);
    EXPECT_EQ(classify_off_pool_allocation(rv, false, 0, false), alloc_category::kv_variable) << "kv_variable (2) ranks above network_output (3)";
    EXPECT_EQ(classify_off_pool_allocation(rv, false, 0, true), alloc_category::kv_variable);
    rv.set_output(false);
}

// ----------------------------------------------------------------------------------------------------------------------------------
// Switch off: nothing is recorded and the allocation sequence is the same as with the switch on.
// ----------------------------------------------------------------------------------------------------------------------------------
TEST(alloc_trace_summary, switch_off_records_nothing_and_does_not_change_allocations) {
    auto& eng = get_test_engine();
    struct observation {
        alloc_trace_summary summary;
        size_t record_count = 0;
        std::vector<std::string> sequence;                // per node: id, output size, allocation type
        std::vector<std::vector<std::string>> share_groups;  // nodes grouped by output buffer_ptr (only groups with >1 node), sorted
        std::vector<std::string> outputs;                 // output values
        int64_t delta = 0;
    };
    auto build_and_measure = [&](int level, observation& o) {
        layout la{ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx};
        layout lb{ov::PartialShape{1, 3, 2, 2}, data_types::f32, format::bfyx};
        topology t;
        t.add(input_layout("A", la));
        t.add(input_layout("B", lb));
        t.add(activation("pa", input_info("A"), activation_func::relu));
        t.add(activation("pb", input_info("B"), activation_func::relu));
        t.add(concatenation("c", {input_info("pa"), input_info("pb")}, 1));
        t.add(activation("m1", input_info("c"), activation_func::abs));
        t.add(activation("m2", input_info("m1"), activation_func::abs));
        t.add(activation("m3", input_info("m2"), activation_func::abs));
        t.add(activation("out", input_info("m3"), activation_func::abs));
        const auto before = eng.get_used_device_memory(allocation_type::usm_device) + eng.get_used_device_memory(allocation_type::usm_host);
        auto prog = program::build_program(eng, t, trace_cfg(eng, level), false, false);
        network net(prog, 0);
        auto ma = eng.allocate_memory(la);
        auto mb = eng.allocate_memory(lb);
        set_values<float>(ma, {1, -2, 3, -4, 5, -6, 7, -8});
        set_values<float>(mb, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        net.set_input_data("A", ma);
        net.set_input_data("B", mb);
        auto outs = net.execute();
        const auto after = eng.get_used_device_memory(allocation_type::usm_device) + eng.get_used_device_memory(allocation_type::usm_host);
        o.delta = static_cast<int64_t>(after) - static_cast<int64_t>(before);
        std::map<const void*, std::vector<std::string>> by_ptr;
        for (const auto& id : net.get_executed_primitive_ids()) {
            auto inst = net.get_primitive(id);
            o.sequence.push_back(id + ":" + std::to_string(inst->output_memory_ptr() ? inst->output_memory().size() : 0) + ":" +
                                 std::to_string(inst->output_memory_ptr() ? static_cast<int>(inst->output_memory().get_allocation_type()) : -1));
            if (inst->output_memory_ptr()) {
                by_ptr[inst->output_memory().buffer_ptr()].push_back(id);
            }
        }
        for (auto& kv : by_ptr) {
            if (kv.second.size() > 1) {
                std::sort(kv.second.begin(), kv.second.end());
                o.share_groups.push_back(kv.second);
            }
        }
        std::sort(o.share_groups.begin(), o.share_groups.end());
        mem_lock<float> l(outs.at("out").get_memory(), get_test_stream());
        for (size_t i = 0; i < 20; i++) {
            o.outputs.push_back(std::to_string(l[i]));
        }
        o.summary = net.get_alloc_trace_summary();
        o.record_count = net.get_alloc_trace_records().size();
    };
    observation off, l1, l2;
    ASSERT_NO_FATAL_FAILURE(build_and_measure(0, off));
    ASSERT_NO_FATAL_FAILURE(build_and_measure(1, l1));
    ASSERT_NO_FATAL_FAILURE(build_and_measure(2, l2));
    EXPECT_EQ(off.summary.total_count, 0u);
    EXPECT_EQ(off.record_count, 0u);
    // level 1 keeps aggregates only, level 2 also stores the per-allocation records
    EXPECT_GT(l1.summary.total_count, 0u);
    EXPECT_EQ(l1.record_count, 0u);
    EXPECT_GT(l2.record_count, 0u);
    EXPECT_EQ(l2.record_count, l2.summary.total_count);
    EXPECT_EQ(l1.summary.total_count, l2.summary.total_count);
    for (const auto* on : {&l1, &l2}) {
        EXPECT_EQ(off.sequence, on->sequence);
        EXPECT_EQ(off.delta, on->delta);
        EXPECT_EQ(off.outputs, on->outputs);
        // block sharing relation (which nodes write into the same buffer) is identical with the switch on and off
        EXPECT_EQ(off.share_groups, on->share_groups);
    }
    EXPECT_FALSE(off.share_groups.empty()) << "expected in-place concat sharing and/or pool reuse in this graph";
}


// ==================================================================================================================================
// Held-bytes snapshot (network::get_held_bytes_snapshot): unique buffers by handle, one bucket each, owner rule for aliases.
// ==================================================================================================================================
namespace {

const held_buffer* find_held(const held_bytes_snapshot& s, const primitive_id& owner) {
    for (const auto& b : s.buffers) {
        if (b.owner == owner) {
            return &b;
        }
    }
    return nullptr;
}

std::string dump_held(const held_bytes_snapshot& s) {
    std::string out;
    for (const auto& b : s.buffers) {
        out += "  owner=" + b.owner + " bucket=" + b.bucket + " bytes=" + std::to_string(b.bytes) + "\n";
    }
    out += "  total=" + std::to_string(s.total_bytes) + " blocks=" + std::to_string(s.total_blocks) + "\n";
    return out;
}

uint64_t sum_buffers(const held_bytes_snapshot& s) {
    uint64_t b = 0;
    for (const auto& x : s.buffers) {
        b += x.bytes;
    }
    return b;
}

}  // namespace

// In-place concat group: predecessors share the holder buffer and are counted once, under the holder.
TEST(held_bytes_snapshot, inplace_concat_group_is_counted_once_and_inputs_are_external) {
    auto& eng = get_test_engine();
    layout la{ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx};
    layout lb{ov::PartialShape{1, 3, 2, 2}, data_types::f32, format::bfyx};
    topology t;
    t.add(input_layout("A", la));
    t.add(input_layout("B", lb));
    t.add(activation("pa", input_info("A"), activation_func::relu));
    t.add(activation("pb", input_info("B"), activation_func::relu));
    t.add(concatenation("c", {input_info("pa"), input_info("pb")}, 1));
    t.add(activation("out", input_info("c"), activation_func::abs));
    auto prog = program::build_program(eng, t, trace_cfg(eng, 0), false, false);
    ASSERT_TRUE(prog->get_node("c").can_be_optimized());
    network net(prog, 0);

    auto before = net.get_held_bytes_snapshot();
    std::cout << dump_held(before);
    EXPECT_EQ(before.external.blocks, 0u) << "inputs are not bound yet";

    auto ma = eng.allocate_memory(la);
    auto mb = eng.allocate_memory(lb);
    set_values<float>(ma, std::vector<float>(8, 1.f));
    set_values<float>(mb, std::vector<float>(12, 2.f));
    net.set_input_data("A", ma);
    net.set_input_data("B", mb);
    net.execute();
    const auto s = net.get_held_bytes_snapshot();
    std::cout << dump_held(s);

    // concat group: one block, owner is the holder, bytes == holder bytes (predecessor views are not added)
    EXPECT_EQ(s.at_off_pool(alloc_category::inplace_concat_group).blocks, 1u);
    EXPECT_EQ(s.at_off_pool(alloc_category::inplace_concat_group).bytes, bytes_of(net, "c"));
    ASSERT_NE(find_held(s, "c"), nullptr);
    EXPECT_EQ(find_held(s, "pa"), nullptr);
    EXPECT_EQ(find_held(s, "pb"), nullptr);
    // external: the two bound inputs
    EXPECT_EQ(s.external.blocks, 2u);
    EXPECT_EQ(s.external.bytes, ma->size() + mb->size());
    // network output
    EXPECT_EQ(s.at_off_pool(alloc_category::network_output).blocks, 1u);
    EXPECT_EQ(s.network_output_bytes, bytes_of(net, "out"));
    // reconciliation with the known constructions
    EXPECT_EQ(s.total_bytes, bytes_of(net, "c") + bytes_of(net, "out") + ma->size() + mb->size());
    EXPECT_EQ(s.total_bytes, sum_buffers(s));
    EXPECT_EQ(s.total_blocks, s.buffers.size());
    EXPECT_EQ(s.ghost.blocks, 0u);
}

// A view (cbo reshape) aliases its producer's buffer: counted once, in the owner's category.
TEST(held_bytes_snapshot, view_is_counted_once_under_the_owner_category) {
    auto& eng = get_test_engine();
    layout l{ov::PartialShape{1, 3, 4, 4}, data_types::f32, format::bfyx};
    topology t;
    t.add(input_layout("A", l));
    t.add(activation("p", input_info("A"), activation_func::relu));
    t.add(reshape("r", input_info("p"), false, std::vector<int64_t>{1, 12, 4, 1}, ov::PartialShape{1, 12, 4, 1}));
    t.add(activation("out", input_info("r"), activation_func::abs));
    auto prog = program::build_program(eng, t, trace_cfg(eng, 0), false, false);
    ASSERT_TRUE(prog->get_node("r").can_be_optimized()) << "reshape must be a view";
    // make the owner non-pool so that the owner category is distinguishable from `pooled`
    prog->get_node("p").can_share_buffer(false);
    network net(prog, 0);
    const auto s = net.get_held_bytes_snapshot();
    std::cout << dump_held(s);
    const auto* p = find_held(s, "p");
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->bucket, std::string(to_string(alloc_category::sum_root)));
    EXPECT_EQ(find_held(s, "r"), nullptr) << "the view is not an owner";
    EXPECT_EQ(s.at_off_pool(alloc_category::sum_root).blocks, 1u);
    EXPECT_EQ(s.at_off_pool(alloc_category::sum_root).bytes, bytes_of(net, "p"));
    EXPECT_EQ(s.total_bytes, bytes_of(net, "p") + bytes_of(net, "out"));
}

// Pool blocks shared by several nodes are counted once per block.
TEST(held_bytes_snapshot, shared_pool_block_is_counted_once) {
    auto& eng = get_test_engine();
    layout l{ov::PartialShape{1, 3, 4, 4}, data_types::f32, format::bfyx};
    topology t;
    t.add(input_layout("A", l));
    t.add(softmax("a1", input_info("A"), 1));
    t.add(softmax("a2", input_info("a1"), 1));
    t.add(softmax("a3", input_info("a2"), 1));
    t.add(softmax("a4", input_info("a3"), 1));
    t.add(softmax("out", input_info("a4"), 1));
    auto prog = program::build_program(eng, t, trace_cfg(eng, 0), false, false);
    network net(prog, 0);
    // intermediates: every executed primitive except the input and the output
    std::set<const void*> unique;
    size_t intermediates = 0;
    for (const auto& id : net.get_executed_primitive_ids()) {
        if (id == "A" || id == "out") {
            continue;
        }
        auto inst = net.get_primitive(id);
        if (inst->output_memory_ptr() && !inst->can_be_optimized()) {
            unique.insert(inst->output_memory().get_internal_params().mem);
            intermediates++;
        }
    }
    const auto s = net.get_held_bytes_snapshot();
    std::cout << dump_held(s);
    ASSERT_GE(intermediates, 2u);
    ASSERT_LT(unique.size(), intermediates) << "expected the pool to reuse a block between non-overlapping intermediates";
    EXPECT_EQ(s.pooled.blocks, unique.size());
    EXPECT_EQ(s.pooled.bytes, unique.size() * bytes_of(net, "out"));
    EXPECT_EQ(s.ghost.blocks, 0u);
    EXPECT_EQ(s.total_bytes, s.pooled.bytes + bytes_of(net, "out"));
}

// const bucket (data node) and external bucket (bound input).
TEST(held_bytes_snapshot, const_and_external_buckets) {
    auto& eng = get_test_engine();
    layout l{ov::PartialShape{1, 3, 4, 4}, data_types::f32, format::bfyx};
    auto w = eng.allocate_memory(l);
    set_values<float>(w, std::vector<float>(48, 1.f));
    topology t;
    t.add(input_layout("A", l));
    t.add(data("w", w));
    t.add(eltwise("sum", {input_info("A"), input_info("w")}, eltwise_mode::sum));
    t.add(activation("out", input_info("sum"), activation_func::abs));
    auto prog = program::build_program(eng, t, trace_cfg(eng, 0), false, false);
    network net(prog, 0);
    auto a = eng.allocate_memory(l);
    set_values<float>(a, std::vector<float>(48, 2.f));
    net.set_input_data("A", a);
    net.execute();
    const auto s = net.get_held_bytes_snapshot();
    std::cout << dump_held(s);
    EXPECT_EQ(s.constant.blocks, 1u);
    EXPECT_EQ(s.constant.bytes, w->size());
    EXPECT_EQ(s.external.blocks, 1u);
    EXPECT_EQ(s.external.bytes, a->size());
    EXPECT_EQ(s.total_bytes, w->size() + a->size() + bytes_of(net, "sum") + bytes_of(net, "out"));
}

// A pool record that no primitive references is reported as ghost.
TEST(held_bytes_snapshot, unreferenced_pool_record_is_reported_as_ghost) {
    auto& eng = get_test_engine();
    layout l{ov::PartialShape{1, 3, 4, 4}, data_types::f32, format::bfyx};
    topology t;
    t.add(input_layout("A", l));
    t.add(activation("p", input_info("A"), activation_func::relu));
    t.add(activation("out", input_info("p"), activation_func::abs));
    auto prog = program::build_program(eng, t, trace_cfg(eng, 0), false, false);
    network net(prog, 0);
    EXPECT_EQ(net.get_held_bytes_snapshot().ghost.blocks, 0u);
    // a block taken from the pool by an id that is not a primitive of the network and kept alive only by the pool record
    memory_restricter<uint32_t> none;
    auto stray = net.get_memory_pool().get_memory(layout{ov::PartialShape{1, 1, 1, 1000}, data_types::f32, format::bfyx}, "stray", 1000000, net.get_id(),
                                                  none, allocation_type::usm_device);
    const auto s = net.get_held_bytes_snapshot();
    stray.reset();
    std::cout << dump_held(s);
    EXPECT_EQ(s.ghost.blocks, 1u);
    EXPECT_GE(s.ghost.bytes, 4000u);
}

// oneDNN sum chain: the writer aliases the chain root; counted once under sum_root.
#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(held_bytes_snapshot, sum_writer_alias_is_counted_once_under_the_root) {
    auto& eng = get_test_engine();
    if (!eng.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN sum post-op requires supports_immad";
    }
    auto make_w = [&](int64_t ofm, int64_t ifm) {
        auto m = eng.allocate_memory(layout{ov::PartialShape{ofm, ifm}, data_types::f32, format::bfyx});
        mem_lock<float> l(m, get_test_stream());
        for (size_t i = 0; i < l.size(); i++) {
            l[i] = 0.01f;
        }
        return m;
    };
    topology t;
    t.add(input_layout("in", layout{ov::PartialShape{32, 64}, data_types::f32, format::bfyx}));
    for (const char* id : {"x_proj", "fc_1", "p"}) {
        t.add(data(std::string(id) + "_w", make_w(64, 64)));
    }
    t.add(fully_connected("x_proj", input_info("in"), "x_proj_w"));
    t.add(fully_connected("fc_1", input_info("x_proj"), "fc_1_w"));
    t.add(fully_connected("p", input_info("fc_1"), "p_w"));
    t.add(eltwise("res", input_info("p"), input_info("x_proj"), eltwise_mode::sum));
    t.add(softmax("out", input_info("res"), 1));
    auto prog = program::build_program(eng, t, trace_cfg(eng, 0), false, false);
    if (prog->get_node("p").get_preferred_impl_type() != impl_types::onednn || prog->get_node("x_proj").can_share_buffer()) {
        GTEST_SKIP() << "oneDNN in-place sum was not selected";
    }
    network net(prog, 0);
    const auto s = net.get_held_bytes_snapshot();
    std::cout << dump_held(s);
    const auto* root = find_held(s, "x_proj");
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->bucket, std::string(to_string(alloc_category::sum_root)));
    EXPECT_EQ(find_held(s, "p"), nullptr) << "the sum writer aliases the root buffer";
    EXPECT_EQ(s.at_off_pool(alloc_category::sum_root).blocks, 1u);
    EXPECT_EQ(s.at_off_pool(alloc_category::sum_root).bytes, bytes_of(net, "x_proj"));
}
#else
TEST(held_bytes_snapshot, sum_writer_alias_is_counted_once_under_the_root) {
    GTEST_SKIP() << "built without oneDNN";
}
#endif

// Dynamic network: the snapshot follows the current holdings (it does not accumulate over reallocations).
TEST(held_bytes_snapshot, dynamic_network_follows_current_holdings) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("input", layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}));
    t.add(activation("a", input_info("input"), activation_func::relu));
    t.add(activation("b", input_info("a"), activation_func::abs));
    t.add(activation("out", input_info("b"), activation_func::relu));
    auto prog = program::build_program(eng, t, trace_cfg(eng, 1), false, false);
    network net(prog, 0);
    auto run = [&](const ov::Shape& sh) {
        auto m = eng.allocate_memory(layout{ov::PartialShape(sh), data_types::f32, format::bfyx});
        set_values<float>(m, std::vector<float>(ov::shape_size(sh), 1.f));
        net.set_input_data("input", m);
        net.execute();
        return net.get_held_bytes_snapshot();
    };
    // Expected holdings (f32): a tensor of shape [1,C,H,W] has C*H*W*4 bytes. Each round holds one pool block, one
    // network-output buffer and the user-bound input. The pool block and the output buffer are dynamic outputs: they grow with the shape and
    // are NOT reclaimed when the shape shrinks, because the default output_reclaim_min_size (32 MiB) is above these sizes (the reclaim
    // rule only releases when the new size is below 1/10 and the buffer exceeds that minimum). The input is user memory and follows the bound tensor.
    struct round_expect {
        ov::Shape shape;
        uint64_t pooled, output, input;
    };
    const uint64_t small_b = 4 * 8 * 8 * 4, big_b = 16 * 64 * 64 * 4;
    const std::vector<round_expect> rounds = {
        {{1, 4, 8, 8}, small_b, small_b, small_b},   // first round: small buffers
        {{1, 16, 64, 64}, big_b, big_b, big_b},      // growth: realloc, the superseded small buffers are gone from the snapshot
        {{1, 16, 64, 64}, big_b, big_b, big_b},      // same shape: nothing changes (no accumulation)
        {{1, 4, 8, 8}, big_b, big_b, small_b},       // shrink: buffers kept (below output_reclaim_min_size), only the input follows
    };
    uint64_t events_before = 0;
    for (size_t i = 0; i < rounds.size(); ++i) {
        const auto& r = rounds[i];
        const auto sn = run(r.shape);
        SCOPED_TRACE("round " + std::to_string(i));
        std::cout << dump_held(sn);
        EXPECT_EQ(sn.pooled.blocks, 1u);
        EXPECT_EQ(sn.pooled.bytes, r.pooled);
        EXPECT_EQ(sn.external.blocks, 1u);
        EXPECT_EQ(sn.external.bytes, r.input);
        EXPECT_EQ(sn.at_off_pool(alloc_category::network_output).blocks, 1u);
        EXPECT_EQ(sn.at_off_pool(alloc_category::network_output).bytes, r.output);
        EXPECT_EQ(sn.ghost.blocks, 0u);
        EXPECT_EQ(sn.total_blocks, 3u);
        EXPECT_EQ(sn.total_bytes, r.pooled + r.output + r.input);
        // cumulative allocation events (not a held quantity) only ever grow; they exceed the held non-input bytes once buffers were superseded
        const auto ev = net.get_alloc_trace_summary().total_bytes;
        EXPECT_GE(ev, events_before);
        events_before = ev;
        if (i >= 1) {
            EXPECT_GT(ev, sn.total_bytes - sn.external.bytes) << "events include the superseded small buffers";
        }
    }
}

// variable bucket: a read_value/assign graph keeps its state in variable memory, which is not an off-pool category.
TEST(held_bytes_snapshot, variable_bucket) {
    auto& eng = get_test_engine();
    layout vl{ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx};
    auto in = eng.allocate_memory(vl);
    set_values<float>(in, std::vector<float>(8, 1.f));
    topology t;
    t.add(input_layout("input", vl));
    t.add(read_value{"read_value", {input_info("input")}, "v0", {vl}});
    t.add(eltwise{"sum", {input_info("input"), input_info("read_value")}, eltwise_mode::sum, {}, vl.data_type});
    t.add(assign{"assign", {input_info("sum")}, "v0", vl});
    auto net = get_network(eng, t, trace_cfg(eng, 0), get_test_stream_ptr(), false);
    auto context = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{eng.get_device()});
    auto variable = std::make_shared<ov::intel_gpu::VariableState>(ov::intel_gpu::VariableStateInfo{"v0", vl}, context, net->get_shape_predictor());
    net->set_variable("v0", variable);
    net->set_input_data("input", in);
    net->execute();
    const auto s = net->get_held_bytes_snapshot();
    std::cout << dump_held(s);
    EXPECT_GE(s.variable.blocks, 1u);
    EXPECT_GT(s.variable.bytes, 0u);
    const auto* rv = find_held(s, "read_value");
    if (rv != nullptr) {
        EXPECT_EQ(rv->bucket, "variable");
    }
    EXPECT_EQ(s.at_off_pool(alloc_category::kv_variable).blocks, 0u) << "variable memory is its own bucket, not an off-pool category";
    // per-buffer attribution fields used by the [held_buf] lines
    for (const auto& b : s.buffers) {
        if (b.bucket == "ghost") {
            continue;
        }
        EXPECT_FALSE(b.owner_type.empty()) << b.owner;
        if (!b.owner_intermediate && b.bucket == "variable") {
            EXPECT_EQ(b.layout_bytes, static_cast<int64_t>(vl.bytes_count())) << b.owner;
            EXPECT_EQ(b.owner_port, 0u) << b.owner;
        }
    }
    if (rv != nullptr) {
        EXPECT_EQ(rv->owner_uid, net->get_program()->get_node("read_value").get_unique_id());
    }
}

#else  // GPU_DEBUG_CONFIG

TEST(alloc_trace_summary, requires_gpu_debug_caps) {
    GTEST_SKIP() << "allocation trace is only available with ENABLE_GPU_DEBUG_CAPS";
}

#endif  // GPU_DEBUG_CONFIG
