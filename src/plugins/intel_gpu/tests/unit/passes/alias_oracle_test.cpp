// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Phase 0.5b-3: runtime oracle of the alias analysis (V0 axis, V2 structural sharing, V3 realized alias edges).
// Design: /home/ceciliapeng/repos/reports/memman/layer1/13_p05b_design.zh-CN.md (section 3.4, Appendix B1 / B3 / B4).
//
// Negative controls (B4): every rule has a case in which the graph is tampered with after the first execute() (a view re-pointed to a
// private buffer, an output re-pointed into a pool block, an injected axis shifted by one) and the oracle is re-run through
// network::run_alias_oracle_for_test(); the same graph before the tamper must report no defect.

#include "test_utils.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "alias_analysis.h"
#include "alias_oracle.h"
#include "exec_axis.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/plugin/output_memory_block.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/condition.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/selective_ssm.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "primitive_inst.h"
#include "program_node.h"

using namespace cldnn;
using namespace ::tests;

namespace {

ExecutionConfig make_cfg(engine& eng, QueueTypes q = QueueTypes::in_order, bool shadow = true) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(q));
    config.set_property(ov::intel_gpu::enable_alias_shadow_validation(shadow));
    if (q == QueueTypes::out_of_order) {
        // T-OOOQ: oneDNN forces the in-order queue on Arc; switch it off by a *user* property
        config.set_user_property({ov::intel_gpu::use_onednn(false)}, ov::OptionVisibility::RELEASE_INTERNAL);
    }
    return config;
}

memory::ptr filled(engine& eng, const layout& l, float seed = 1.f) {
    auto m = eng.allocate_memory(l);
    std::vector<float> v(l.count());
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = seed + static_cast<float>(i % 13) * 0.25f - 1.5f;
    }
    if (l.data_type == data_types::f32) {
        set_values(m, v);
    } else if (l.data_type == data_types::f16) {
        std::vector<ov::float16> h(v.begin(), v.end());
        set_values(m, h);
    }
    return m;
}

// only the logical elements of the output: the buffer of a dynamic output is larger than its layout (the tail is not initialized)
std::vector<float> read_f32(const network_output& o) {
    auto m = o.get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> p(m, get_test_stream());
    return std::vector<float>(p.begin(), p.begin() + std::min<size_t>(p.size(), o.get_layout().count()));
}

layout l4(int64_t c = 16, data_types dt = data_types::f32) {
    return layout{ov::PartialShape{1, c, 8, 8}, dt, format::bfyx};
}

// reorder -> in-place concat -> reorder, plus a crop view and an independent chain
topology concat_crop_topology() {
    topology t;
    t.add(input_layout("in1", l4()));
    t.add(input_layout("in2", l4()));
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(reorder("out", input_info("C"), format::bfyx, data_types::f32));
    t.add(input_layout("in3", l4()));
    t.add(activation("B1", input_info("in3"), activation_func::relu));
    t.add(crop("K", input_info("B1"), tensor(1, 8, 8, 8), tensor(0, 0, 0, 0)));
    t.add(activation("outK", input_info("K"), activation_func::abs));
    return t;
}

void set_concat_crop_inputs(engine& eng, network& net) {
    net.set_input_data("in1", filled(eng, l4(), 1.f));
    net.set_input_data("in2", filled(eng, l4(), 2.f));
    net.set_input_data("in3", filled(eng, l4(), 3.f));
}

// two independent chains, the last node of each is a network output; the middle ones live in the pool
topology two_chains_topology() {
    topology t;
    t.add(input_layout("in1", l4(4)));
    t.add(input_layout("in2", l4(4)));
    t.add(activation("a1", input_info("in1"), activation_func::relu));
    t.add(activation("a2", input_info("a1"), activation_func::abs));
    t.add(activation("oa", input_info("a2"), activation_func::abs));
    t.add(activation("b1", input_info("in2"), activation_func::relu));
    t.add(activation("b2", input_info("b1"), activation_func::abs));
    t.add(activation("ob", input_info("b2"), activation_func::abs));
    return t;
}

void set_two_chain_inputs(engine& eng, network& net) {
    net.set_input_data("in1", filled(eng, l4(4), 1.f));
    net.set_input_data("in2", filled(eng, l4(4), 2.f));
}

std::string dump(const alias_oracle_report& r) {
    std::ostringstream os;
    os << "ran=" << r.ran << " skipped=" << r.skipped_reason << " axis=" << static_cast<int>(r.axis_status) << " v0=" << r.v0_checked
       << " v2_pairs=" << r.v2_pairs << " by_class=" << r.v2_by_class << " by_pool=" << r.v2_by_pool << " unverifiable=" << r.v2_unverifiable
       << " excluded=" << r.v2_excluded << " v3=" << r.v3_checked << " v3_unverifiable=" << r.v3_unverifiable << "\n";
    for (const auto& h : r.hits) {
        os << "  " << to_string(h.id) << (h.sev == severity::defect ? "/D " : "/I ") << h.a << ":" << h.port_a << " " << h.b << ":" << h.port_b << " " << h.detail << "\n";
    }
    return os.str();
}

#define ASSERT_REPORT(net)                                              \
    ASSERT_NE((net).get_alias_oracle_report(), nullptr);                \
    const alias_oracle_report& rep = *(net).get_alias_oracle_report()

bool has_hit(const alias_oracle_report& r, rule_id id, severity sev, const std::string& a = "", const std::string& b = "") {
    for (const auto& h : r.hits) {
        if (h.id == id && h.sev == sev && (a.empty() || h.a == a || h.b == a) && (b.empty() || h.a == b || h.b == b)) {
            return true;
        }
    }
    return false;
}

program::ptr roundtrip(engine& eng, program& prog, const ExecutionConfig& cfg) {
    cldnn::membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        cldnn::BinaryOutputBuffer ob = cldnn::BinaryOutputBuffer(out_mem);
        ob.set_stream(get_test_stream_ptr().get());
        prog.save(ob);
    }
    std::istream in_mem(&mem_buf);
    cldnn::BinaryInputBuffer ib = cldnn::BinaryInputBuffer(in_mem, eng);
    auto imported = std::make_shared<cldnn::program>(eng, cfg);
    imported->load(ib);
    return imported;
}

}  // namespace

// ==============================================================================================================================
// O1. static graph: concat slots, crop view, pool: no defect, V3 and V0 actually checked.
// ==============================================================================================================================
TEST(alias_oracle, O1_static_chain_concat_crop_is_clean) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, concat_crop_topology(), make_cfg(eng), false, false);
    network net(prog, 0);
    set_concat_crop_inputs(eng, net);
    net.execute();
    ASSERT_REPORT(net);
    EXPECT_TRUE(rep.ran) << dump(rep);
    EXPECT_TRUE(rep.skipped_reason.empty()) << dump(rep);
    EXPECT_EQ(rep.defect_count(), 0u) << dump(rep);
    EXPECT_GT(rep.v3_checked, 0u) << dump(rep);
    EXPECT_EQ(rep.axis_status, exec_axis::status::ok);
    EXPECT_EQ(rep.v0_checked, net.get_executed_primitive_ids().size());
    // sampled once: a second round does not change the report
    const auto before = dump(rep);
    net.set_alias_oracle_fault_for_test(true);   // would turn the report into an exception report if the oracle ran again
    net.execute();
    EXPECT_EQ(dump(*net.get_alias_oracle_report()), before);
}

// V3 negative control: a realized edge is broken after the first round
TEST(alias_oracle, O1b_V3_detects_a_member_that_left_its_provider) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, concat_crop_topology(), make_cfg(eng), false, false);
    network net(prog, 0);
    set_concat_crop_inputs(eng, net);
    net.execute();
    const auto* aa = prog->get_alias_analysis();
    ASSERT_NE(aa, nullptr);
    net.run_alias_oracle_for_test();
    ASSERT_EQ(net.get_alias_oracle_report()->defect_count(), 0u) << dump(*net.get_alias_oracle_report());
    const alias_edge* victim = nullptr;
    for (const auto& e : aa->edges()) {
        const auto inst = net.get_primitive(e.member_node->id());
        if (e.act == edge_activity::ALWAYS && inst && !inst->get_flag(ExecutionFlags::SKIP) && inst->output_memory_ptr(e.member_port)) {
            victim = &e;
            break;
        }
    }
    ASSERT_NE(victim, nullptr);
    auto inst = net.get_primitive(victim->member_node->id());
    inst->set_output_memory(eng.allocate_memory(inst->output_memory_ptr(victim->member_port)->get_layout()), false, victim->member_port);
    net.run_alias_oracle_for_test();
    const auto& rep = *net.get_alias_oracle_report();
    EXPECT_GE(rep.count(rule_id::V3, severity::defect), 1u) << dump(rep);
    EXPECT_TRUE(has_hit(rep, rule_id::V3, severity::defect, victim->member_node->id())) << dump(rep);
}

// V0 negative control: an injected axis that is shifted by one
TEST(alias_oracle, O1c_V0_detects_a_wrong_axis) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, concat_crop_topology(), make_cfg(eng), false, false);
    network net(prog, 0);
    set_concat_crop_inputs(eng, net);
    net.execute();
    const auto real = exec_axis::build(*prog);
    ASSERT_TRUE(real.valid());
    net.run_alias_oracle_for_test(&real);
    EXPECT_EQ(net.get_alias_oracle_report()->count(rule_id::V0, severity::defect), 0u);
    std::unordered_map<const program_node*, uint32_t> shifted;
    for (auto* n : prog->get_processing_order()) {
        if (real.position(*n)) {
            shifted[n] = *real.position(*n) + 1;
        }
    }
    const auto bad = exec_axis::for_test(exec_axis::status::ok, shifted);
    net.run_alias_oracle_for_test(&bad);
    const auto& rep = *net.get_alias_oracle_report();
    EXPECT_GT(rep.v0_checked, 0u);
    EXPECT_EQ(rep.count(rule_id::V0, severity::defect), rep.v0_checked) << dump(rep);
}

// ==============================================================================================================================
// O2. V2: sharing of two classes must be explained by the pool. Negative control: an output re-pointed into a pool block.
// ==============================================================================================================================
TEST(alias_oracle, O2_V2_detects_unexplained_sharing) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, two_chains_topology(), make_cfg(eng), false, false);
    network net(prog, 0);
    set_two_chain_inputs(eng, net);
    net.execute();
    ASSERT_REPORT(net);
    EXPECT_EQ(rep.defect_count(), 0u) << dump(rep);
    // control: the network really has pool sharing, explained by the pool or by a class
    net.run_alias_oracle_for_test();
    EXPECT_EQ(net.get_alias_oracle_report()->defect_count(), 0u) << dump(*net.get_alias_oracle_report());
    // tamper: the output "ob" now lives in the buffer of "a1" (a pool user of another class, never registered for ob)
    auto a1 = net.get_primitive("a1");
    auto ob = net.get_primitive("ob");
    ASSERT_NE(a1->output_memory_ptr(0), nullptr);
    ob->set_output_memory(a1->output_memory_ptr(0), false, 0);
    net.run_alias_oracle_for_test();
    const auto& rep2 = *net.get_alias_oracle_report();
    EXPECT_TRUE(has_hit(rep2, rule_id::V2, severity::defect, "ob", "a1")) << dump(rep2);
}

// F27 detector: selective_ssm re-points output 1 to input 5 on an empty sequence, without any alias edge
TEST(alias_oracle, O2b_V2_reports_the_selective_ssm_state_alias_F27) {
    auto& eng = get_test_engine();
    const int64_t heads = 2, groups = 1, head_dim = 4, state_size = 8;
    topology t;
    t.add(input_layout("A", layout{ov::PartialShape{heads}, data_types::f32, format::bfyx}));
    t.add(input_layout("dt", layout{ov::PartialShape{-1, -1, heads}, data_types::f32, format::bfyx}));
    t.add(input_layout("B", layout{ov::PartialShape{-1, -1, groups, state_size}, data_types::f32, format::bfyx}));
    t.add(input_layout("x", layout{ov::PartialShape{-1, -1, heads, head_dim}, data_types::f32, format::bfyx}));
    t.add(input_layout("C", layout{ov::PartialShape{-1, -1, groups, state_size}, data_types::f32, format::bfyx}));
    t.add(input_layout("state_in", layout{ov::PartialShape{-1, heads, head_dim, state_size}, data_types::f32, format::bfyx}));
    t.add(activation("state", input_info("state_in"), activation_func::relu));   // input 5 is produced by a node, not by an input_layout
    t.add(selective_ssm("selective_ssm", {input_info("A"), input_info("dt"), input_info("B"), input_info("x"), input_info("C"), input_info("state")}));
    t.add(reorder("output", input_info("selective_ssm", 0), format::bfyx, data_types::f32));
    t.add(reorder("state_output", input_info("selective_ssm", 1), format::bfyx, data_types::f32));
    ExecutionConfig cfg = make_cfg(eng);
    program::ptr prog;
    try {
        prog = program::build_program(eng, t, cfg, false, false);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "selective_ssm graph cannot be built here: " << e.what();
    }
    network net(prog, 0);
    net.set_input_data("A", filled(eng, layout{ov::PartialShape{heads}, data_types::f32, format::bfyx}));
    net.set_input_data("dt", eng.allocate_memory(layout{ov::PartialShape{1, 0, heads}, data_types::f32, format::bfyx}));
    net.set_input_data("B", eng.allocate_memory(layout{ov::PartialShape{1, 0, groups, state_size}, data_types::f32, format::bfyx}));
    net.set_input_data("x", eng.allocate_memory(layout{ov::PartialShape{1, 0, heads, head_dim}, data_types::f32, format::bfyx}));
    net.set_input_data("C", eng.allocate_memory(layout{ov::PartialShape{1, 0, groups, state_size}, data_types::f32, format::bfyx}));
    net.set_input_data("state_in", filled(eng, layout{ov::PartialShape{1, heads, head_dim, state_size}, data_types::f32, format::bfyx}));
    net.execute();
    ASSERT_REPORT(net);
    EXPECT_TRUE(rep.ran) << dump(rep);
    // selective_ssm itself is skipped (SKIP without being cbo / runtime skippable: excluded, Appendix B3.5); its consumer "state_output" is the
    // cbo view of port 1, whose class is not the class of "state", yet both sit in the same buffer
    EXPECT_TRUE(has_hit(rep, rule_id::V2, severity::defect, "state", "state_output") || has_hit(rep, rule_id::V2, severity::defect, "state", "selective_ssm"))
        << "F27: output 1 lives in the buffer of input 5 without an alias edge\n" << dump(rep);
}

// ==============================================================================================================================
// O3 / O5. P1-B-S / P1-B-D switches, dynamic MAY members that allocate themselves in some rounds: no false positive.
// ==============================================================================================================================
TEST(alias_oracle, O3_concat_holder_pool_static_and_dynamic_no_false_positive) {
    auto& eng = get_test_engine();
    for (const bool pool : {false, true}) {
        auto cfg = make_cfg(eng);
        cfg.set_property(ov::intel_gpu::enable_concat_holder_pool(pool));
        cfg.set_property(ov::intel_gpu::enable_dynamic_concat_holder_pool(pool));
        {   // static
            auto prog = program::build_program(eng, concat_crop_topology(), cfg, false, false);
            network net(prog, 0);
            set_concat_crop_inputs(eng, net);
            net.execute();
            ASSERT_NE(net.get_alias_oracle_report(), nullptr);
            EXPECT_EQ(net.get_alias_oracle_report()->defect_count(), 0u) << "pool=" << pool << "\n" << dump(*net.get_alias_oracle_report());
        }
        {   // dynamic runtime concat, rounds that fuse and that do not (a zero-length predecessor): sampled after every round
            topology t;
            const layout dyn{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
            t.add(input_layout("in1", dyn));
            t.add(input_layout("in2", dyn));
            t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
            t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
            t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
            t.add(reorder("output", input_info("C"), format::bfyx, data_types::f32));
            auto prog = program::build_program(eng, t, cfg, false, false);
            network net(prog, 0);
            const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> rounds{{{1, 8, 3, 3}, {1, 4, 3, 3}}, {{1, 2, 5, 5}, {1, 0, 5, 5}}, {{1, 6, 4, 4}, {1, 6, 4, 4}}};
            for (const auto& r : rounds) {
                net.set_input_data("in1", filled(eng, layout{ov::PartialShape(r.first), data_types::f32, format::bfyx}, 1.f));
                net.set_input_data("in2", filled(eng, layout{ov::PartialShape(r.second), data_types::f32, format::bfyx}, 2.f));
                net.execute();
                net.run_alias_oracle_for_test();
                EXPECT_EQ(net.get_alias_oracle_report()->defect_count(), 0u) << "pool=" << pool << "\n" << dump(*net.get_alias_oracle_report());
            }
        }
    }
}

// ==============================================================================================================================
// O4. sharing explained by the pool: a long chain reuses blocks
// ==============================================================================================================================
TEST(alias_oracle, O4_pool_sharing_is_explained_by_the_pool) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", l4(4)));
    std::string prev = "in";
    for (int i = 1; i <= 8; ++i) {
        t.add(activation("n" + std::to_string(i), input_info(prev), i % 2 ? activation_func::relu : activation_func::abs));
        prev = "n" + std::to_string(i);
    }
    t.add(activation("out", input_info(prev), activation_func::abs));
    auto cfg = make_cfg(eng);
    cfg.set_property(ov::intel_gpu::optimize_data(false));   // no activation fusing: every node of the chain owns a pool buffer
    auto prog = program::build_program(eng, t, cfg, false, false);
    network net(prog, 0);
    net.set_input_data("in", filled(eng, l4(4)));
    net.execute();
    ASSERT_REPORT(net);
    EXPECT_EQ(rep.defect_count(), 0u) << dump(rep);
    EXPECT_GE(rep.v2_by_pool, 1u) << "the chain must reuse pool blocks\n" << dump(rep);
}

// ==============================================================================================================================
// O6 / O7 / O8 / O9
// ==============================================================================================================================
TEST(alias_oracle, O6_imported_program_is_skipped_with_a_reason) {
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng);
    auto prog = program::build_program(eng, concat_crop_topology(), cfg, false, false);
    auto imported = roundtrip(eng, *prog, cfg);
    EXPECT_EQ(imported->get_alias_analysis(), nullptr);
    network net(imported, 0);
    set_concat_crop_inputs(eng, net);
    net.execute();
    ASSERT_REPORT(net);
    EXPECT_FALSE(rep.ran);
    EXPECT_EQ(rep.skipped_reason, "no_analysis");
}

TEST(alias_oracle, O7_out_of_order_queue_runs_the_structural_part_only) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, two_chains_topology(), make_cfg(eng, QueueTypes::out_of_order), false, false);
    ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::out_of_order);
    network net(prog, 0);
    set_two_chain_inputs(eng, net);
    net.execute();
    ASSERT_REPORT(net);
    EXPECT_TRUE(rep.ran) << dump(rep);
    EXPECT_EQ(rep.axis_status, exec_axis::status::not_in_order);
    EXPECT_EQ(rep.v0_checked, 0u);
    EXPECT_EQ(rep.defect_count(), 0u) << dump(rep);
}

TEST(alias_oracle, O8_switch_off_has_no_reports) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, two_chains_topology(), make_cfg(eng, QueueTypes::in_order, /*shadow=*/false), false, false);
    EXPECT_EQ(prog->get_alias_validation_report(), nullptr);
    network net(prog, 0);
    set_two_chain_inputs(eng, net);
    net.execute();
    EXPECT_EQ(net.get_alias_oracle_report(), nullptr);
}

TEST(alias_oracle, O9_two_networks_of_one_program_have_independent_reports) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, concat_crop_topology(), make_cfg(eng), false, false);
    network n1(prog, 0);
    network n2(prog, 1);
    set_concat_crop_inputs(eng, n1);
    set_concat_crop_inputs(eng, n2);
    n1.execute();
    n2.execute();
    ASSERT_NE(n1.get_alias_oracle_report(), nullptr);
    ASSERT_NE(n2.get_alias_oracle_report(), nullptr);
    EXPECT_NE(n1.get_alias_oracle_report(), n2.get_alias_oracle_report());
    EXPECT_EQ(n1.get_alias_oracle_report()->defect_count(), 0u);
    EXPECT_EQ(n2.get_alias_oracle_report()->defect_count(), 0u);
    // tamper only n1
    const auto* aa = prog->get_alias_analysis();
    const alias_edge* e = nullptr;
    for (const auto& ed : aa->edges()) {
        const auto inst = n1.get_primitive(ed.member_node->id());
        if (ed.act == edge_activity::ALWAYS && inst && !inst->get_flag(ExecutionFlags::SKIP) && inst->output_memory_ptr(ed.member_port)) {
            e = &ed;
            break;
        }
    }
    ASSERT_NE(e, nullptr);
    auto inst = n1.get_primitive(e->member_node->id());
    inst->set_output_memory(eng.allocate_memory(inst->output_memory_ptr(e->member_port)->get_layout()), false, e->member_port);
    n1.run_alias_oracle_for_test();
    n2.run_alias_oracle_for_test();
    EXPECT_GE(n1.get_alias_oracle_report()->defect_count(), 1u);
    EXPECT_EQ(n2.get_alias_oracle_report()->defect_count(), 0u);
}

// ==============================================================================================================================
// B1: the oracle must not break execute(): an exception inside the oracle leaves the per-round flags reset
// ==============================================================================================================================
TEST(alias_oracle, B1_exception_in_the_oracle_does_not_leak_into_execute_and_flags_are_reset) {
    auto& eng = get_test_engine();
    auto make_net = [&](bool shadow) {
        topology t;
        const layout dyn{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
        t.add(input_layout("in", dyn));
        t.add(activation("a", input_info("in"), activation_func::relu));
        t.add(activation("b", input_info("a"), activation_func::abs));
        t.add(activation("out", input_info("b"), activation_func::abs));
        return std::make_shared<network>(program::build_program(eng, t, make_cfg(eng, QueueTypes::in_order, shadow), false, false), 0);
    };
    auto ref = make_net(false);
    auto net = make_net(true);
    ASSERT_NE(net->get_alias_oracle_report(), nullptr);
    // the raw oracle does throw when the fault is armed (the guard in execute() is what keeps it away from the caller)
    net->set_alias_oracle_fault_for_test(true);
    EXPECT_ANY_THROW(net->run_alias_oracle_for_test());
    net->set_alias_oracle_fault_for_test(true);
    for (const std::vector<int64_t>& shape : {std::vector<int64_t>{1, 4, 3, 3}, std::vector<int64_t>{1, 6, 5, 5}}) {
        const layout l{ov::PartialShape(shape), data_types::f32, format::bfyx};
        auto in = filled(eng, l);
        ref->set_input_data("in", in);
        net->set_input_data("in", in);
        auto ref_out = ref->execute();
        std::map<primitive_id, network_output> out;
        EXPECT_NO_THROW(out = net->execute());
        EXPECT_EQ(read_f32(out.at("out")), read_f32(ref_out.at("out")));
        // every flag of every instance of the round has been reset
        for (const auto& id : net->get_executed_primitive_ids()) {
            const auto inst = net->get_primitive(id);
            for (size_t f : {ExecutionFlags::SHAPE_CHANGED, ExecutionFlags::IMPL_CHANGED, ExecutionFlags::MEMORY_CHANGED, ExecutionFlags::ARG_UPDATE_REQUIRED, ExecutionFlags::SKIP}) {
                EXPECT_FALSE(inst->get_flag(f)) << id << " flag " << f << " still set after the oracle threw";
            }
        }
        if (shape[1] == 4) {
            ASSERT_NE(net->get_alias_oracle_report(), nullptr);
            EXPECT_FALSE(net->get_alias_oracle_report()->ran);
            EXPECT_EQ(net->get_alias_oracle_report()->skipped_reason.find("exception"), 0u) << net->get_alias_oracle_report()->skipped_reason;
        }
    }
    // not retried: the report still describes the failed first sampling
    EXPECT_EQ(net->get_alias_oracle_report()->skipped_reason.find("exception"), 0u);
}

// ==============================================================================================================================
// V2 false positive guards (B3): assign, condition pass-through, external output block
// ==============================================================================================================================
TEST(alias_oracle, G1_assign_does_not_report_a_defect) {
    auto& eng = get_test_engine();
    const layout vl{data_types::f32, format::bfyx, tensor{1, 1, 1, 3}};
    auto input_data = eng.allocate_memory(vl);
    set_values(input_data, std::vector<float>{1.f, 2.f, 3.f});
    topology t;
    t.add(input_layout("input", vl));
    t.add(read_value{"read_value", {input_info("input")}, "v0", {vl}});
    t.add(eltwise{"sum", {input_info("input"), input_info("read_value")}, eltwise_mode::sum, {}, vl.data_type});
    t.add(assign{"assign", {input_info("sum")}, "v0", vl});
    auto prog = program::build_program(eng, t, make_cfg(eng), false, false);
    network net(prog, 0);
    auto context = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{eng.get_device()});
    auto variable = std::make_shared<ov::intel_gpu::VariableState>(ov::intel_gpu::VariableStateInfo{"v0", vl}, context, net.get_shape_predictor());
    net.set_variable("v0", variable);
    net.set_input_data("input", input_data);
    for (int i = 0; i < 2; ++i) {
        net.execute();
        net.run_alias_oracle_for_test();
        ASSERT_NE(net.get_alias_oracle_report(), nullptr);
        EXPECT_EQ(net.get_alias_oracle_report()->defect_count(), 0u) << dump(*net.get_alias_oracle_report());
    }
}

TEST(alias_oracle, G2_condition_pass_through_branch_is_not_a_defect) {
    auto& eng = get_test_engine();
    const layout dl{data_types::f32, format::bfyx, tensor{1, 1, 4, 1}};
    const layout pl{data_types::u8, format::bfyx, tensor{1, 1, 1, 1}};
    auto build = [&](bool shadow) -> std::shared_ptr<network> {
        ExecutionConfig cfg = make_cfg(eng, QueueTypes::in_order, shadow);
        auto branch = [&](bool pass_through, const std::string& in_id, const std::string& out_id) {
            topology b;
            b.add(input_layout(in_id, dl));
            if (pass_through) {
                b.add(reorder(out_id, input_info(in_id), dl.format, dl.data_type));   // identity: optimized out, the branch output is its input
            } else {
                b.add(activation(out_id, input_info(in_id), activation_func::abs));
            }
            condition::branch br;
            br.inner_program = program::build_program(eng, b, cfg, false, false, true);
            br.input_map.insert({"input", in_id});
            br.output_map.insert({0, out_id});
            return br;
        };
        topology t;
        t.add(input_layout("predicate", pl));
        t.add(input_layout("input", dl));
        t.add(condition("cond", {input_info("predicate"), input_info("input")}, branch(true, "bi_t", "o_t"), branch(false, "bi_f", "o_f")));
        t.add(activation("tail", input_info("cond"), activation_func::abs));
        return std::make_shared<network>(program::build_program(eng, t, cfg, false, false), 0);
    };
    auto run = [&](network& net, uint8_t p) {
        auto pred = eng.allocate_memory(pl);
        set_values(pred, std::vector<uint8_t>{p});
        auto in = eng.allocate_memory(dl);
        set_values(in, std::vector<float>{1.f, -2.f, 3.f, -4.f});
        net.set_input_data("predicate", pred);
        net.set_input_data("input", in);
        net.execute();
    };
    // probe with the switch off: if a pass-through branch does not execute at all, that is independent of the oracle
    try {
        auto probe = build(false);
        run(*probe, 1);
        run(*probe, 0);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "a pass-through condition branch does not run with the switch off either (" << e.what() << "): construction not usable";
    }
    auto net = build(true);
    for (const uint8_t p : {uint8_t{1}, uint8_t{0}, uint8_t{1}}) {
        run(*net, p);
        net->run_alias_oracle_for_test();
        ASSERT_NE(net->get_alias_oracle_report(), nullptr);
        EXPECT_EQ(net->get_alias_oracle_report()->defect_count(), 0u) << "predicate=" << int(p) << "\n" << dump(*net->get_alias_oracle_report());
    }
}

TEST(alias_oracle, G3_external_output_block_is_not_a_defect) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", layout{ov::PartialShape{-1, 8}, data_types::f32, format::bfyx}));
    t.add(activation("a", input_info("in"), activation_func::relu));
    t.add(activation("out", input_info("a"), activation_func::abs));
    auto prog = program::build_program(eng, t, make_cfg(eng), false, false);
    network net(prog, 0);
    ov::intel_gpu::OutputMemoryBlock block(eng);
    net.register_output_memory_block("out", &block);
    for (const int64_t m : {3, 5, 2}) {
        net.set_input_data("in", filled(eng, layout{ov::PartialShape{m, 8}, data_types::f32, format::bfyx}));
        net.execute();
        net.run_alias_oracle_for_test();
        ASSERT_NE(net.get_alias_oracle_report(), nullptr);
        EXPECT_EQ(net.get_alias_oracle_report()->defect_count(), 0u) << "m=" << m << "\n" << dump(*net.get_alias_oracle_report());
    }
    net.clear_output_memory_blocks();
}

// ==============================================================================================================================
// AB. neutrality (design section 6): enable_alias_shadow_validation on / off in the same binary: memory dependency string, flag and
// restriction snapshot, allocation trace and outputs must be identical. Run with the P1-B-S / P1-B-D switches off and on (N7): only then does
// the class partition influence the allocation.
// ==============================================================================================================================
namespace {
struct ab_result {
    std::string memdeps;
    std::string flags;
    std::string trace;
    std::vector<std::vector<float>> outputs;
};

ab_result ab_run(engine& eng, const topology& topo, bool shadow, bool pool, bool dyn_pool, const std::function<void(network&, int)>& feed, int rounds,
                 const std::vector<std::string>& output_ids) {
    ExecutionConfig cfg = make_cfg(eng, QueueTypes::in_order, shadow);
    cfg.set_property(ov::intel_gpu::enable_concat_holder_pool(pool));
    cfg.set_property(ov::intel_gpu::enable_dynamic_concat_holder_pool(dyn_pool));
    cfg.set_user_property({ov::intel_gpu::dump_alloc_trace(2)});
    auto prog = program::build_program(eng, topo, cfg, false, false);
    ab_result r;
    std::ostringstream md;   // same content as program::get_memory_dependencies_string() (private): node, uid, restricted uids
    std::ostringstream fl;
    for (auto* n : prog->get_processing_order()) {
        md << "primitive: " << n->id() << "(unique_id:" << n->get_unique_id() << ") restricted list: ";
        for (auto uid : n->get_memory_dependencies()) {
            md << uid << ",";
        }
        md << "\n";
        fl << n->id() << " cbo=" << n->can_be_optimized() << " rs=" << n->is_runtime_skippable() << " out=" << n->is_output() << " cbs=" << n->can_share_buffer()
           << " dyn=" << n->is_dynamic() << " deps=";
        for (auto uid : n->get_memory_dependencies()) {
            fl << uid << ",";
        }
        fl << "\n";
    }
    r.memdeps = md.str();
    r.flags = fl.str();
    network net(prog, 0);
    for (int i = 0; i < rounds; ++i) {
        feed(net, i);
        auto outs = net.execute();
        for (const auto& id : output_ids) {
            r.outputs.push_back(read_f32(outs.at(id)));
        }
    }
#ifdef GPU_DEBUG_CONFIG
    std::ostringstream tr;
    for (const auto& rec : net.get_alloc_trace_records()) {
        tr << rec.node_id << ":" << rec.port << " " << rec.bytes << " pooled=" << rec.pooled << " " << rec.reason << "\n";
    }
    r.trace = tr.str();
#endif
    if (shadow) {
        EXPECT_NE(net.get_alias_oracle_report(), nullptr);
        EXPECT_NE(prog->get_alias_validation_report(), nullptr);
    } else {
        EXPECT_EQ(net.get_alias_oracle_report(), nullptr);
        EXPECT_EQ(prog->get_alias_validation_report(), nullptr);
    }
    return r;
}

void expect_neutral(engine& eng, const topology& topo, const std::function<void(network&, int)>& feed, int rounds, const std::vector<std::string>& outs,
                    const std::string& tag) {
    for (const auto& sw : std::vector<std::pair<bool, bool>>{{false, false}, {true, false}, {true, true}}) {
        const auto off = ab_run(eng, topo, false, sw.first, sw.second, feed, rounds, outs);
        const auto off2 = ab_run(eng, topo, false, sw.first, sw.second, feed, rounds, outs);
        const auto on = ab_run(eng, topo, true, sw.first, sw.second, feed, rounds, outs);
        const std::string t = tag + " pool=" + std::to_string(sw.first) + " dyn_pool=" + std::to_string(sw.second);
        // the baseline itself must be reproducible, otherwise a difference says nothing about the switch
        EXPECT_EQ(off2.outputs, off.outputs) << t << ": the baseline (switch off) is not reproducible";
        EXPECT_EQ(on.memdeps, off.memdeps) << t;
        EXPECT_EQ(on.flags, off.flags) << t;
        EXPECT_EQ(on.trace, off.trace) << t;
        ASSERT_EQ(on.outputs.size(), off.outputs.size()) << t;
        for (size_t i = 0; i < on.outputs.size(); ++i) {
            EXPECT_EQ(on.outputs[i], off.outputs[i]) << t << " output " << i;   // bitwise: same floats
        }
        EXPECT_FALSE(on.flags.empty());
#ifdef GPU_DEBUG_CONFIG
        EXPECT_FALSE(on.trace.empty()) << t << ": allocation trace expected";
#endif
    }
}
}  // namespace

TEST(alias_oracle, AB_static_graphs_are_neutral) {
    auto& eng = get_test_engine();
    expect_neutral(eng, concat_crop_topology(), [&](network& n, int) { set_concat_crop_inputs(eng, n); }, 2, {"out", "outK"}, "concat_crop");
    expect_neutral(eng, two_chains_topology(), [&](network& n, int) { set_two_chain_inputs(eng, n); }, 2, {"oa", "ob"}, "two_chains");
}

TEST(alias_oracle, AB_dynamic_concat_is_neutral) {
    auto& eng = get_test_engine();
    topology t;
    const layout dyn{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    t.add(input_layout("in1", dyn));
    t.add(input_layout("in2", dyn));
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(reorder("output", input_info("C"), format::bfyx, data_types::f32));
    const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> rounds{{{1, 8, 3, 3}, {1, 4, 3, 3}}, {{1, 2, 5, 5}, {1, 0, 5, 5}}, {{1, 6, 4, 4}, {1, 6, 4, 4}}};
    expect_neutral(eng, t, [&](network& n, int i) {
        n.set_input_data("in1", filled(eng, layout{ov::PartialShape(rounds[i].first), data_types::f32, format::bfyx}, 1.f));
        n.set_input_data("in2", filled(eng, layout{ov::PartialShape(rounds[i].second), data_types::f32, format::bfyx}, 2.f));
    }, static_cast<int>(rounds.size()), {"output"}, "dynamic_concat");
}
