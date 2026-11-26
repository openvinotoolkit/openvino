// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// P2-shadow: oooq gate shadow experiment (observation only). T1..T12 of the task brief.
// Question under test: if oooq_memory_dependencies were skipped for in-order queues, which pairs lose their restriction (D = S1 \ S0) and does
// the in-order lifetime oracle (alias classes + exec_axis) call any of them dangerous?
// Hand graphs are built with no_optimizations, flags are set by the test. Nothing here depends on oneDNN.

#include "test_utils.h"

#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "alias_analysis.h"
#include "exec_axis.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "oooq_gate_shadow.h"
#include "pass_manager.h"
#include "program_node.h"
#include "program_wrapper.h"
#include "runtime_concat_order.h"

using namespace cldnn;
using namespace ::tests;

namespace {

struct shadow_on {
    explicit shadow_on(int mode = 1) {
        oooq_gate_shadow::clear_reports_for_test();
        oooq_gate_shadow::set_enabled_for_test(mode);
        oooq_gate_shadow::set_s0_erase_for_test({});
    }
    ~shadow_on() {
        oooq_gate_shadow::set_enabled_for_test(-1);
        oooq_gate_shadow::set_s0_erase_for_test({});
    }
};

ExecutionConfig make_cfg(engine& eng, QueueTypes q, const std::vector<std::string>& outputs) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(q));
    if (!outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(outputs));
    }
    if (q == QueueTypes::out_of_order) {
        // T-OOOQ: oneDNN forces the in-order queue; it has to be switched off by a *user* property.
        config.set_user_property({ov::intel_gpu::use_onednn(false)}, ov::OptionVisibility::RELEASE_INTERNAL);
    }
    return config;
}

program::ptr make_hand(engine& eng, const topology& t, const std::vector<std::string>& outs, QueueTypes q = QueueTypes::in_order) {
    auto prog = program::build_program(eng, t, make_cfg(eng, q, outs), false, /*no_optimizations=*/true);
    for (auto* n : prog->get_processing_order()) {
        n->get_output_layouts(false);
        n->set_unique_id();
    }
    return prog;
}

void force_order(program& prog, const std::vector<primitive_id>& ids) {
    auto& po = prog.get_processing_order();
    for (size_t i = ids.size() - 1; i-- > 0;) {
        auto& node = prog.get_node(ids[i]);
        auto& next = prog.get_node(ids[i + 1]);
        po.erase(&node);
        po.insert(&next, &node);
    }
}

layout f32_l() {
    return layout{ov::PartialShape{1, 4, 8, 8}, data_types::f32, format::bfyx};
}

std::shared_ptr<const oooq_gate_shadow_report> run(program& prog) {
    program_wrapper::prepare_memory_dependencies(prog);
    return oooq_gate_shadow::report_for(prog);
}

using id_pair = std::pair<std::string, std::string>;
id_pair norm(const std::string& a, const std::string& b) {
    return a < b ? id_pair{a, b} : id_pair{b, a};
}
std::set<id_pair> d_set(const oooq_gate_shadow_report& r) {
    std::set<id_pair> s;
    for (const auto& p : r.d_pair_ids) {
        s.insert(norm(p.first, p.second));
    }
    return s;
}
std::string dump(const oooq_gate_shadow_report& r) {
    std::string s = r.summary() + "\n D: ";
    for (const auto& p : d_set(r)) {
        s += "(" + p.first + "," + p.second + ") ";
    }
    for (const auto& p : r.danger_pairs) {
        s += "\n DANGER (" + p.a + "," + p.b + ") " + p.why + (p.in_s1 ? " in_S1" : "");
    }
    for (const auto& p : r.danger_asym_pairs) {
        s += "\n ASYM (" + p.a + "," + p.b + ") " + p.why;
    }
    return s;
}

// in -> P -> {X_k -> Y_k}, k = 0..2; J1 = Y0 + Y1; J = J1 + Y2 -> O
topology pattern_a_topology() {
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    for (int k = 0; k < 3; ++k) {
        t.add(activation("X" + std::to_string(k), input_info("P"), activation_func::relu));
        t.add(activation("Y" + std::to_string(k), input_info("X" + std::to_string(k)), activation_func::abs));
    }
    t.add(eltwise("J1", {input_info("Y0"), input_info("Y1")}, eltwise_mode::sum));
    t.add(eltwise("J", {input_info("J1"), input_info("Y2")}, eltwise_mode::sum));
    t.add(activation("O", input_info("J"), activation_func::abs));
    return t;
}
const std::vector<primitive_id> pattern_a_order = {"in", "P", "X0", "Y0", "X1", "Y1", "J1", "X2", "Y2", "J", "O"};

// A -> X -> B, A -> u, B -> u  (multi-input join)
topology diamond_topology() {
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("A", input_info("in"), activation_func::relu));
    t.add(activation("X", input_info("A"), activation_func::relu));
    t.add(activation("B", input_info("X"), activation_func::abs));
    t.add(eltwise("u", {input_info("A"), input_info("B")}, eltwise_mode::sum));
    t.add(activation("O", input_info("u"), activation_func::abs));
    return t;
}

}  // namespace

// T1. Pattern A: three sibling branches with a shared upstream. oooq restricts the structurally unreachable siblings; on the in-order axis
// (depth-first order) the lifetimes of the early siblings are over before the later ones start.
TEST(oooq_gate_shadow, T1_pattern_a_siblings_are_in_D_and_safe) {
    shadow_on on;
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, pattern_a_topology(), {"O"});
    force_order(*prog, pattern_a_order);
    auto r = run(*prog);
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->ran) << r->mode;
    std::cout << dump(*r) << std::endl;
    EXPECT_EQ(r->mode, "in_order");
    EXPECT_EQ(r->danger, 0u) << dump(*r);
    EXPECT_EQ(r->danger_in_d, 0u);
    EXPECT_EQ(r->danger_s1, 0u) << dump(*r);
    EXPECT_EQ(r->unverifiable, 0u);
    EXPECT_EQ(r->po_exec_mismatch, 0u);
    // structurally unreachable pairs of different branches whose lifetimes are disjoint and that skipped_branch / basic do not restrict
    // (P, Y2): P is dead after X2 and Y2 is born later, but X0 (a user of P) is unrelated to Y2: oooq's "fork" rule
    const std::set<id_pair> expected = {norm("P", "Y2"), norm("X0", "X1"), norm("X0", "Y1"), norm("X0", "X2"), norm("X0", "Y2"), norm("Y0", "X2"),
                                        norm("Y0", "Y2"), norm("X1", "X2"), norm("X1", "Y2"), norm("Y1", "X2"), norm("Y1", "Y2")};
    EXPECT_EQ(d_set(*r), expected) << dump(*r);
    EXPECT_EQ(r->d, expected.size());
    EXPECT_EQ(r->d_unreachable, expected.size() - 1);
    EXPECT_EQ(r->d_fork, 1u);
    EXPECT_EQ(r->d_gain_pairs, expected.size());
    EXPECT_GT(r->d_gain_bytes, 0u);
    // the sibling relaxation is a real gain on the axis: S1 is a strict superset of S0
    EXPECT_GT(r->s1, r->s0);
}

// T2. Plain chain: oooq adds nothing over basic + skipped_branch.
TEST(oooq_gate_shadow, T2_chain_has_empty_D) {
    shadow_on on;
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("A", input_info("in"), activation_func::relu));
    t.add(activation("B", input_info("A"), activation_func::abs));
    t.add(activation("C", input_info("B"), activation_func::abs));
    t.add(activation("O", input_info("C"), activation_func::abs));
    auto prog = make_hand(eng, t, {"O"});
    auto r = run(*prog);
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->ran);
    EXPECT_EQ(r->d, 0u) << dump(*r);
    EXPECT_EQ(r->danger, 0u) << dump(*r);
    EXPECT_EQ(r->danger_s1, 0u) << dump(*r);
    EXPECT_EQ(r->unverifiable, 0u);
}

// T2b. Residual chain A -> B -> C, R = A + C.
TEST(oooq_gate_shadow, T2b_residual_has_no_danger) {
    shadow_on on;
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("A", input_info("in"), activation_func::relu));
    t.add(activation("B", input_info("A"), activation_func::abs));
    t.add(activation("C", input_info("B"), activation_func::abs));
    t.add(eltwise("R", {input_info("A"), input_info("C")}, eltwise_mode::sum));
    t.add(activation("O", input_info("R"), activation_func::abs));
    auto prog = make_hand(eng, t, {"O"});
    auto r = run(*prog);
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->ran);
    EXPECT_EQ(r->d, 0u) << dump(*r);
    EXPECT_EQ(r->danger, 0u) << dump(*r);
    EXPECT_EQ(r->danger_s1, 0u) << dump(*r);
}

// T3. Multi-input join A -> X -> B, A -> u, B -> u: no alias through the join, no lifetime missed.
TEST(oooq_gate_shadow, T3_multi_input_join_no_danger) {
    shadow_on on;
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, diamond_topology(), {"O"});
    force_order(*prog, {"in", "A", "X", "B", "u", "O"});
    auto r = run(*prog);
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->ran);
    std::cout << dump(*r) << std::endl;
    EXPECT_EQ(r->danger, 0u) << dump(*r);
    EXPECT_EQ(r->danger_in_d, 0u);
    EXPECT_EQ(r->danger_s1, 0u) << dump(*r);
    EXPECT_EQ(r->unverifiable, 0u);
    EXPECT_EQ(r->d, 0u) << dump(*r);
}

// T4. suspect_nodes section: u = A + B has two non-constant dependencies with A -> ... -> B; oooq then restricts A against every user of
// B (here v), a pair no other rule covers. In-order lifetime of A ends at u, v is born after u: the pair is a real gain, not a danger.
TEST(oooq_gate_shadow, T4_suspect_section_pairs_are_in_D_and_safe) {
    shadow_on on;
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("A", input_info("in"), activation_func::relu));
    t.add(activation("X", input_info("A"), activation_func::relu));
    t.add(activation("B", input_info("X"), activation_func::abs));
    t.add(eltwise("u", {input_info("A"), input_info("B")}, eltwise_mode::sum));
    t.add(eltwise("v", {input_info("B"), input_info("u")}, eltwise_mode::sum));
    t.add(activation("O", input_info("v"), activation_func::abs));
    auto prog = make_hand(eng, t, {"O"});
    force_order(*prog, {"in", "A", "X", "B", "u", "v", "O"});
    auto r = run(*prog);
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->ran);
    std::cout << dump(*r) << std::endl;
    EXPECT_TRUE(r->in_d("A", "v")) << dump(*r);
    EXPECT_GE(r->d_suspect, 1u) << dump(*r);
    EXPECT_EQ(r->d_suspect_danger, 0u);
    EXPECT_EQ(r->danger, 0u) << dump(*r);
    EXPECT_EQ(r->danger_s1, 0u) << dump(*r);
}

// T5. View (cbo crop) whose reader is late: the owner P is read through K after M has been born. The legacy folding of views keeps (M, P)
// restricted in S0 already, so the gate cannot lose it.
TEST(oooq_gate_shadow, T5_view_reader_late_no_danger) {
    shadow_on on;
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(crop("K", input_info("P"), tensor(1, 4, 8, 8), tensor(0, 0, 0, 0)));
    t.add(activation("E", input_info("P"), activation_func::abs));
    t.add(activation("M", input_info("E"), activation_func::abs));
    t.add(activation("R", input_info("K"), activation_func::abs));
    t.add(eltwise("O", {input_info("M"), input_info("R")}, eltwise_mode::sum));
    auto prog = make_hand(eng, t, {"O"});
    prog->get_node("K").can_be_optimized(true);
    force_order(*prog, {"in", "P", "K", "E", "M", "R", "O"});
    auto r = run(*prog);
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->ran);
    std::cout << dump(*r) << std::endl;
    EXPECT_EQ(r->danger, 0u) << dump(*r);
    EXPECT_EQ(r->danger_s1, 0u) << dump(*r);
    EXPECT_EQ(r->unverifiable, 0u);
    EXPECT_GE(r->same_class_pairs, 1u);   // P and K share a class by design
}

// T6. Static in-place concat: the whole group (holder + predecessors) is one class with the lifetime [first predecessor, last reader of C].
TEST(oooq_gate_shadow, T6_inplace_concat_group_no_danger) {
    shadow_on on;
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", f32_l()));
    t.add(input_layout("in2", f32_l()));
    t.add(input_layout("in3", layout{ov::PartialShape{1, 8, 8, 8}, data_types::f32, format::bfyx}));
    t.add(activation("P1", input_info("in1"), activation_func::relu));
    t.add(activation("Z", input_info("in3"), activation_func::relu));
    t.add(activation("P2", input_info("in2"), activation_func::abs));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1, data_types::f32));
    t.add(activation("Q", input_info("C"), activation_func::abs));
    t.add(eltwise("O", {input_info("Q"), input_info("Z")}, eltwise_mode::sum));
    auto prog = make_hand(eng, t, {"O"});
    prog->get_node("C").can_be_optimized(true);
    // what prepare_buffer_fusing does to the predecessors of an in-place concat
    prog->get_node("P1").can_share_buffer(false);
    prog->get_node("P2").can_share_buffer(false);
    force_order(*prog, {"in1", "in2", "in3", "P1", "Z", "P2", "C", "Q", "O"});
    auto r = run(*prog);
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->ran);
    std::cout << dump(*r) << std::endl;
    EXPECT_EQ(r->danger, 0u) << dump(*r);
    EXPECT_EQ(r->danger_s1, 0u) << dump(*r);
    EXPECT_EQ(r->unverifiable, 0u);
    EXPECT_GE(r->same_class_pairs, 2u);   // C-P1 and C-P2
}

namespace {
topology dyn_branch_topology() {
    auto dyn = [] { return layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}; };
    topology t;
    t.add(input_layout("in1", dyn()));
    t.add(input_layout("in2", dyn()));
    t.add(eltwise("D", {input_info("in1"), input_info("in1")}, eltwise_mode::sum));
    t.add(eltwise("Z", {input_info("in2"), input_info("in2")}, eltwise_mode::sum));
    t.add(reorder("P0", input_info("D"), format::bfyx, data_types::f16));
    t.add(reorder("P1", input_info("Z"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P0"), input_info("P1")}, 1));
    t.add(reorder("output", input_info("C"), format::bfyx, data_types::f32));
    return t;
}

program::ptr build_to_post(engine& eng, const topology& t, const ExecutionConfig& cfg) {
    auto prog = program::build_program(eng, t, cfg, false, true);
    program_wrapper::pre_optimize_graph(*prog, false);
    program_wrapper::run_graph_compilation(*prog);
    program_wrapper::post_optimize_graph(*prog, false);   // runs prepare_memory_dependencies (and with it the shadow)
    return prog;
}
}  // namespace

// T7. Dynamic runtime concat (F9 moves its predecessors in front of it): PO equals _exec_order and the shadow sees a valid axis.
TEST(oooq_gate_shadow, T7_dynamic_concat_predecessors_po_equals_exec_order) {
    shadow_on on;
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng, QueueTypes::in_order, {});
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = build_to_post(eng, dyn_branch_topology(), cfg);
    ASSERT_TRUE(prog->get_node("C").can_be_optimized());
    ASSERT_TRUE(prog->get_node("C").is_dynamic());
    program_wrapper::prepare_memory_dependencies(*prog);
    auto r = oooq_gate_shadow::report_for(*prog);
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->ran) << r->mode;
    std::cout << dump(*r) << std::endl;
    EXPECT_EQ(r->po_exec_mismatch, 0u);
    EXPECT_EQ(r->danger, 0u) << dump(*r);
    network net(prog, 0);
    const auto ids = net.get_executed_primitive_ids();
    const auto filtered = runtime_concat_order::filtered_processing_order(prog->get_processing_order());
    ASSERT_EQ(ids.size(), filtered.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        EXPECT_EQ(ids[i], filtered[i]->id()) << "index " << i;
    }
}

// T7b. Static network: the same equality.
TEST(oooq_gate_shadow, T7b_static_network_po_equals_exec_order) {
    shadow_on on;
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng, QueueTypes::in_order, {"O"});
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(eng, pattern_a_topology(), cfg, false, false);
    auto r = oooq_gate_shadow::report_for(*prog);
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->ran) << r->mode;
    EXPECT_EQ(r->po_exec_mismatch, 0u);
    EXPECT_EQ(r->danger, 0u) << dump(*r);
    network net(prog, 0);
    const auto ids = net.get_executed_primitive_ids();
    const auto filtered = runtime_concat_order::filtered_processing_order(prog->get_processing_order());
    ASSERT_EQ(ids.size(), filtered.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        EXPECT_EQ(ids[i], filtered[i]->id()) << "index " << i;
    }
}

// T8. The first non-constant node of the processing order never takes part in an oooq pair: the pair loop of oooq_memory_dependencies
// increments its outer index before the inner loop. Observed on a graph: (in1, X) is structurally unreachable and lifetime-disjoint, yet it
// is absent from S1 while the same pair is in S1 \ S0 as soon as in1 is not the first node (control run with a dummy first node).
TEST(oooq_gate_shadow, T8_first_node_gap_actual_behavior) {
    shadow_on on;
    auto& eng = get_test_engine();
    auto make = [&](bool with_dummy_first) {
        topology t;
        if (with_dummy_first) {
            t.add(input_layout("dummy", f32_l()));
        }
        t.add(input_layout("in1", f32_l()));
        t.add(input_layout("in2", f32_l()));
        t.add(activation("A", input_info("in1"), activation_func::relu));
        t.add(activation("X", input_info("in2"), activation_func::relu));
        t.add(activation("Y", input_info("X"), activation_func::abs));
        t.add(eltwise("E", {input_info("A"), input_info("Y")}, eltwise_mode::sum));
        t.add(activation("O", input_info("E"), activation_func::abs));
        auto prog = make_hand(eng, t, {"O"});
        std::vector<primitive_id> order;
        if (with_dummy_first) {
            order.push_back("dummy");
        }
        for (const char* id : {"in1", "A", "in2", "X", "Y", "E", "O"}) {
            order.push_back(id);
        }
        force_order(*prog, order);
        return prog;
    };
    {
        auto prog = make(false);
        auto r = run(*prog);
        ASSERT_NE(r, nullptr);
        std::cout << "[T8 first node = in1] " << dump(*r) << std::endl;
        // in1 is the first node: it is in no oooq pair
        EXPECT_FALSE(prog->get_node("in1").has_memory_dependency(prog->get_node("X").get_unique_id()));
        EXPECT_FALSE(prog->get_node("X").has_memory_dependency(prog->get_node("in1").get_unique_id()));
        EXPECT_EQ(r->d_first_node, 0u) << dump(*r);
        EXPECT_FALSE(r->in_d("in1", "X"));
    }
    {
        auto prog = make(true);
        auto r = run(*prog);
        ASSERT_NE(r, nullptr);
        std::cout << "[T8 control: dummy first] " << dump(*r) << std::endl;
        EXPECT_TRUE(prog->get_node("in1").has_memory_dependency(prog->get_node("X").get_unique_id()));
        EXPECT_TRUE(r->in_d("in1", "X")) << dump(*r);
        EXPECT_EQ(r->danger, 0u) << dump(*r);
    }
}

// T9. Network output class: death = infinity. A plain output node is not a pool user (the allocator bypasses the pool), so the pair
// is not judged (non_pool_pairs). T9b: the output is a cbo view (reorder) of a pooled producer P: the class {P, OUTV} lives forever, and
// the only thing that keeps later nodes off P's block is oooq (C-P2-1): the gate would drop it.
TEST(oooq_gate_shadow, T9_plain_output_node_is_not_a_pool_user) {
    shadow_on on;
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(input_layout("in2", f32_l()));
    t.add(activation("OUT", input_info("in"), activation_func::relu));
    t.add(activation("B1", input_info("in2"), activation_func::relu));
    t.add(activation("B2", input_info("B1"), activation_func::abs));
    t.add(activation("O2", input_info("B2"), activation_func::abs));
    auto prog = make_hand(eng, t, {"OUT", "O2"});
    force_order(*prog, {"in", "OUT", "in2", "B1", "B2", "O2"});
    auto r = run(*prog);
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->ran);
    std::cout << dump(*r) << std::endl;
    EXPECT_EQ(r->danger, 0u) << dump(*r);
    EXPECT_GE(r->non_pool_pairs, 4u);   // OUT against everything else
}

TEST(oooq_gate_shadow, T9b_output_view_class_is_infinite_and_needs_oooq) {
    shadow_on on;
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(input_layout("in2", f32_l()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(reorder("OUTV", input_info("P"), format::bfyx, data_types::f32));
    t.add(activation("B1", input_info("in2"), activation_func::relu));
    t.add(activation("B2", input_info("B1"), activation_func::abs));
    t.add(activation("O2", input_info("B2"), activation_func::abs));
    auto prog = make_hand(eng, t, {"OUTV", "O2"});
    prog->get_node("OUTV").can_be_optimized(true);
    force_order(*prog, {"in", "P", "OUTV", "in2", "B1", "B2", "O2"});
    auto r = run(*prog);
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->ran);
    std::cout << dump(*r) << std::endl;
    bool inf_seen = false;
    for (const auto& p : r->danger_pairs) {
        inf_seen = inf_seen || p.why.find("inf") != std::string::npos;
    }
    // C-P2-1 characterization: today the pairs (P, B1) and (P, B2) are restricted by oooq only (unreachable pairs), so a gate would drop them
    EXPECT_TRUE(inf_seen) << dump(*r);
    EXPECT_GE(r->danger_in_d, 1u) << dump(*r);
    EXPECT_EQ(r->danger_s1, 0u) << dump(*r);   // LEGACY (S1) is complete
    EXPECT_TRUE(r->in_d("P", "B1") || r->in_d("P", "B2")) << dump(*r);
}

namespace {
std::vector<std::pair<std::string, std::vector<uint32_t>>> restrictions(program& prog) {
    std::vector<std::pair<std::string, std::vector<uint32_t>>> v;
    for (auto* n : prog.get_processing_order()) {
        v.emplace_back(n->id(), n->get_memory_dependencies());
    }
    return v;
}
}  // namespace

// T10. Out-of-order queue: the shadow is skipped and counted; restrictions are bit-identical to a run with the switch off.
TEST(oooq_gate_shadow, T10_out_of_order_queue_is_skipped_and_neutral) {
    auto& eng = get_test_engine();
    std::vector<std::pair<std::string, std::vector<uint32_t>>> off, on_r;
    {
        shadow_on off_guard(0);
        auto prog = make_hand(eng, pattern_a_topology(), {"O"}, QueueTypes::out_of_order);
        ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::out_of_order);
        auto r = run(*prog);
        EXPECT_EQ(r, nullptr);
        off = restrictions(*prog);
    }
    {
        shadow_on on(1);
        auto prog = make_hand(eng, pattern_a_topology(), {"O"}, QueueTypes::out_of_order);
        auto r = run(*prog);
        ASSERT_NE(r, nullptr);
        EXPECT_FALSE(r->ran);
        EXPECT_EQ(r->mode, "skipped:oooq");
        on_r = restrictions(*prog);
    }
    EXPECT_EQ(off, on_r);
}

// T11. Switch off vs on: every restriction set is identical (observation only), in-order as well.
TEST(oooq_gate_shadow, T11_switch_does_not_change_restrictions) {
    auto& eng = get_test_engine();
    std::vector<std::pair<std::string, std::vector<uint32_t>>> off, on_r;
    {
        shadow_on guard(0);
        auto prog = make_hand(eng, pattern_a_topology(), {"O"});
        force_order(*prog, pattern_a_order);
        EXPECT_EQ(run(*prog), nullptr);
        off = restrictions(*prog);
    }
    {
        shadow_on guard(1);
        auto prog = make_hand(eng, pattern_a_topology(), {"O"});
        force_order(*prog, pattern_a_order);
        ASSERT_NE(run(*prog), nullptr);
        on_r = restrictions(*prog);
    }
    ASSERT_FALSE(off.empty());
    EXPECT_EQ(off, on_r);
}

// T12. The oracle can fail: removing a legitimately overlapping pair from S0 must be reported as danger (and as danger_in_D because oooq
// or basic still has it in S1).
TEST(oooq_gate_shadow, T12_oracle_detects_a_removed_pair) {
    shadow_on on;
    auto& eng = get_test_engine();
    {
        auto prog = make_hand(eng, diamond_topology(), {"O"});
        force_order(*prog, {"in", "A", "X", "B", "u", "O"});
        auto r = run(*prog);
        ASSERT_NE(r, nullptr);
        ASSERT_EQ(r->danger, 0u) << dump(*r);
    }
    oooq_gate_shadow::set_s0_erase_for_test({{"A", "X"}, {"A", "B"}});
    auto prog = make_hand(eng, diamond_topology(), {"O"});
    force_order(*prog, {"in", "A", "X", "B", "u", "O"});
    auto r = run(*prog);
    ASSERT_NE(r, nullptr);
    std::cout << dump(*r) << std::endl;
    EXPECT_GE(r->danger, 2u) << dump(*r);
    EXPECT_EQ(r->danger_in_d, r->danger);
    EXPECT_GE(r->d, 2u);
    EXPECT_FALSE(r->danger_in_d_pairs.empty());
}

// Switch-off sanity: no report at all and no snapshot work.
TEST(oooq_gate_shadow, T13_disabled_produces_no_report) {
    shadow_on guard(0);
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, diamond_topology(), {"O"});
    program_wrapper::prepare_memory_dependencies(*prog);
    EXPECT_EQ(oooq_gate_shadow::report_for(*prog), nullptr);
}
