// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// P2-a S2: switch, gate computation and neutrality of the restrictions (design 19, L18 / L19 / L20 / L21, gate part).
// In S2 the gate is only computed: the restrictions must be bit-identical with the switch on and off, in-order and out-of-order.
// Hand graphs are built with no_optimizations; nothing depends on oneDNN.

#include "test_utils.h"

#include <string>
#include <utility>
#include <vector>

#include "alias_analysis.h"
#include "class_lifetime.h"
#include "exec_axis.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "pass_manager.h"
#include "program_node.h"
#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

namespace {

layout f32_l() {
    return layout{ov::PartialShape{1, 4, 8, 8}, data_types::f32, format::bfyx};
}

// in -> P -> {X_k -> Y_k}, k = 0..2; J1 = Y0 + Y1; J = J1 + Y2 -> O  (Pattern A siblings) plus a crop view of P
topology graph_topology() {
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(crop("K", input_info("P"), tensor(1, 4, 8, 8), tensor(0, 0, 0, 0)));
    for (int k = 0; k < 3; ++k) {
        t.add(activation("X" + std::to_string(k), input_info("P"), activation_func::relu));
        t.add(activation("Y" + std::to_string(k), input_info("X" + std::to_string(k)), activation_func::abs));
    }
    t.add(eltwise("J1", {input_info("Y0"), input_info("Y1")}, eltwise_mode::sum));
    t.add(eltwise("J", {input_info("J1"), input_info("Y2")}, eltwise_mode::sum));
    t.add(activation("R", input_info("K"), activation_func::abs));
    t.add(eltwise("O", {input_info("J"), input_info("R")}, eltwise_mode::sum));
    return t;
}

program::ptr make_hand(engine& eng, bool lifetime_switch, QueueTypes q, bool internal = false, bool body = false) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(q));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"O"}));
    config.set_property(ov::intel_gpu::enable_inorder_lifetime_restrictions(lifetime_switch));
    if (q == QueueTypes::out_of_order) {
        // T-OOOQ: oneDNN forces the in-order queue unless a *user* property decides
        config.set_user_property({ov::intel_gpu::use_onednn(false)}, ov::OptionVisibility::RELEASE_INTERNAL);
    }
    auto prog = program::build_program(eng, graph_topology(), config, internal, /*no_optimizations=*/true, body);
    for (auto* n : prog->get_processing_order()) {
        n->get_output_layouts(false);
        n->set_unique_id();
    }
    prog->get_node("K").can_be_optimized(true);
    return prog;
}

using restriction_dump = std::vector<std::pair<std::string, std::vector<uint32_t>>>;

restriction_dump run_and_dump(program& prog) {
    program_wrapper::prepare_memory_dependencies(prog);
    restriction_dump v;
    for (auto* n : prog.get_processing_order()) {
        v.emplace_back(n->id(), n->get_memory_dependencies());
    }
    return v;
}

// same text as program::get_memory_dependencies_string() (private)
std::string memdeps_string(program& prog) {
    std::string s = "Memory dependencies/restrictions:\n";
    for (auto* n : prog.get_processing_order()) {
        s += "primitive: " + n->id() + "(unique_id:" + std::to_string(n->get_unique_id()) + ") restricted list: ";
        for (auto uid : n->get_memory_dependencies()) {
            s += std::to_string(uid) + ",";
        }
        s += "\n";
    }
    return s;
}

bool is_ocl_engine(engine& eng) {
    return eng.type() == engine_types::ocl && eng.runtime_type() == runtime_types::ocl;
}

}  // namespace

// S2-1. The switch exists, is release-internal and defaults to off.
TEST(inorder_lifetime, S2_1_switch_defaults_to_off) {
    auto& eng = get_test_engine();
    ExecutionConfig config = get_test_default_config(eng);
    EXPECT_FALSE(config.get_enable_inorder_lifetime_restrictions());
    auto prog = make_hand(eng, false, QueueTypes::in_order);
    const auto r = evaluate_inorder_lifetime_gate(*prog, exec_axis::build(*prog));
    EXPECT_FALSE(r.open);
    EXPECT_STREQ(r.reason, "switch_off");
}

// L18 (gate part). In-order + switch on + OCL + plain program: open; a not normalized axis closes it.
TEST(inorder_lifetime, L18_gate_follows_the_axis) {
    auto& eng = get_test_engine();
    if (!is_ocl_engine(eng)) {
        GTEST_SKIP() << "the gate requires the OCL backend; this test engine is " << eng.type() << " (uncovered)";
    }
    auto prog = make_hand(eng, true, QueueTypes::in_order);
    ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::in_order);
    const auto axis = exec_axis::build(*prog);
    ASSERT_TRUE(axis.valid());
    const auto open = evaluate_inorder_lifetime_gate(*prog, axis);
    EXPECT_TRUE(open.open);
    EXPECT_STREQ(open.reason, "open");
    for (const auto st : {exec_axis::status::not_normalized, exec_axis::status::not_in_order}) {
        const auto closed = evaluate_inorder_lifetime_gate(*prog, exec_axis::for_test(st, {}));
        EXPECT_FALSE(closed.open);
        EXPECT_STREQ(closed.reason, "exec_axis_not_valid");
    }
}

// L19. Out-of-order (oneDNN disabled by a user property, queue asserted): the gate is closed and the restriction strings are bit-identical.
TEST(inorder_lifetime, L19_out_of_order_gate_closed_and_restrictions_identical) {
    auto& eng = get_test_engine();
    restriction_dump off, on;
    std::string off_str, on_str;
    {
        auto prog = make_hand(eng, false, QueueTypes::out_of_order);
        ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::out_of_order);
        off = run_and_dump(*prog);
        off_str = memdeps_string(*prog);
    }
    {
        auto prog = make_hand(eng, true, QueueTypes::out_of_order);
        ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::out_of_order);
        ASSERT_TRUE(prog->get_config().get_enable_inorder_lifetime_restrictions());
        const auto r = evaluate_inorder_lifetime_gate(*prog, exec_axis::build(*prog));
        EXPECT_FALSE(r.open);
        EXPECT_STREQ(r.reason, "queue_not_in_order");
        on = run_and_dump(*prog);
        on_str = memdeps_string(*prog);
    }
    ASSERT_FALSE(off.empty());
    EXPECT_EQ(off, on);
    EXPECT_EQ(off_str, on_str);
}

// L20. A body program is never gated.
TEST(inorder_lifetime, L20_body_program_gate_closed) {
    auto& eng = get_test_engine();
    if (!is_ocl_engine(eng)) {
        GTEST_SKIP() << "non OCL engine: the backend condition is evaluated first (uncovered)";
    }
    auto prog = make_hand(eng, true, QueueTypes::in_order, /*internal=*/false, /*body=*/true);
    ASSERT_TRUE(prog->is_body_program());
    const auto r = evaluate_inorder_lifetime_gate(*prog, exec_axis::build(*prog));
    EXPECT_FALSE(r.open);
    EXPECT_STREQ(r.reason, "body_program");
    const auto off = make_hand(eng, false, QueueTypes::in_order, false, true);
    EXPECT_EQ(run_and_dump(*prog), run_and_dump(*off));
}

// L21. An internal program is never gated.
TEST(inorder_lifetime, L21_internal_program_gate_closed) {
    auto& eng = get_test_engine();
    if (!is_ocl_engine(eng)) {
        GTEST_SKIP() << "non OCL engine: the backend condition is evaluated first (uncovered)";
    }
    auto prog = make_hand(eng, true, QueueTypes::in_order, /*internal=*/true, /*body=*/false);
    ASSERT_TRUE(prog->is_internal_program());
    const auto r = evaluate_inorder_lifetime_gate(*prog, exec_axis::build(*prog));
    EXPECT_FALSE(r.open);
    EXPECT_STREQ(r.reason, "internal_program");
    const auto off = make_hand(eng, false, QueueTypes::in_order, true, false);
    EXPECT_EQ(run_and_dump(*prog), run_and_dump(*off));
}

// S2-2. oooq_memory_dependencies keeps working with an explicit nullptr filter (the plumbing of S4) and with the default argument.
TEST(inorder_lifetime, S2_2_oooq_pass_accepts_a_null_leaf_filter) {
    auto& eng = get_test_engine();
    auto a = make_hand(eng, false, QueueTypes::in_order);
    auto b = make_hand(eng, false, QueueTypes::in_order);
    program_wrapper::prepare_memory_dependencies(*a);
    program_wrapper::prepare_memory_dependencies(*b);
    ASSERT_NO_THROW(oooq_memory_dependencies(static_cast<const inorder_leaf_filter*>(nullptr)).run(*a));
    ASSERT_NO_THROW(oooq_memory_dependencies().run(*b));
    restriction_dump da, db;
    for (auto* n : a->get_processing_order()) {
        da.emplace_back(n->id(), n->get_memory_dependencies());
    }
    for (auto* n : b->get_processing_order()) {
        db.emplace_back(n->id(), n->get_memory_dependencies());
    }
    EXPECT_EQ(da, db);
}
