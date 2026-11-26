// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// P2-a S1: white-box tests of the class_lifetimes engine (design 19, L25 and L12).
// Hand graphs are built with no_optimizations, flags are set by the test. Nothing here depends on oneDNN.

#include "test_utils.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "alias_analysis.h"
#include "class_lifetime.h"
#include "exec_axis.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/selective_ssm.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "program_node.h"

using namespace cldnn;
using namespace ::tests;

namespace {

layout f32_l() {
    return layout{ov::PartialShape{1, 4, 8, 8}, data_types::f32, format::bfyx};
}

program::ptr make_hand(engine& eng, const topology& t, const std::vector<std::string>& outs) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    if (!outs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(outs));
    }
    auto prog = program::build_program(eng, t, config, false, /*no_optimizations=*/true);
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

bool has_node(const std::vector<const program_node*>& v, const program_node& n) {
    return std::find(v.begin(), v.end(), &n) != v.end();
}

// in -> P -> K(crop view of P) ; E = abs(P) ; M = abs(E) ; R = abs(K) ; O = M + R
topology view_topology() {
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(crop("K", input_info("P"), tensor(1, 4, 8, 8), tensor(0, 0, 0, 0)));
    t.add(activation("E", input_info("P"), activation_func::abs));
    t.add(activation("M", input_info("E"), activation_func::abs));
    t.add(activation("R", input_info("K"), activation_func::abs));
    t.add(eltwise("O", {input_info("M"), input_info("R")}, eltwise_mode::sum));
    return t;
}
const std::vector<primitive_id> view_order = {"in", "P", "K", "E", "M", "R", "O"};

}  // namespace

// L25a. A view class lives from its first member to the last reader of ANY member (the reader of the view counts).
TEST(class_lifetime, L25a_view_class_interval_covers_readers_of_the_view) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, view_topology(), {"O"});
    prog->get_node("K").can_be_optimized(true);
    force_order(*prog, view_order);
    auto aa = alias_analysis::build(*prog);
    const auto axis = exec_axis::build(*prog);
    ASSERT_TRUE(axis.valid());
    const auto lt = class_lifetimes::build(*prog, aa, axis);
    ASSERT_TRUE(lt.valid());
    const auto* cp = aa.class_of(prog->get_node("P"), 0);
    ASSERT_NE(cp, nullptr);
    EXPECT_EQ(cp, aa.class_of(prog->get_node("K"), 0));
    const auto& iv = lt.interval_of(*cp);
    ASSERT_TRUE(iv.known);
    EXPECT_FALSE(iv.death_inf);
    EXPECT_EQ(iv.birth, *axis.position(prog->get_node("P")));
    EXPECT_EQ(iv.death, *axis.position(prog->get_node("R"))) << "R reads the view K, it is the last reader of the class";
    EXPECT_GT(iv.death, *axis.position(prog->get_node("E")));
    // classes_disjoint: P/K share a class, P/E do not
    EXPECT_FALSE(lt.classes_disjoint(prog->get_node("P"), prog->get_node("K")));
    EXPECT_TRUE(lt.classes_disjoint(prog->get_node("P"), prog->get_node("E")));
    // the allocator of the class is P (K is a view), E allocates on its own
    EXPECT_TRUE(has_node(lt.pool_allocators(*cp), prog->get_node("P")));
    EXPECT_FALSE(has_node(lt.pool_allocators(*cp), prog->get_node("K")));
}

// L25a'. member_interval: a member on its own lives from its position to the last reader of its own tensor (and of what aliases it).
TEST(class_lifetime, L25a2_member_interval_is_narrower_than_the_class) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, view_topology(), {"O"});
    prog->get_node("K").can_be_optimized(true);
    force_order(*prog, view_order);
    auto aa = alias_analysis::build(*prog);
    const auto axis = exec_axis::build(*prog);
    ASSERT_TRUE(axis.valid());
    const auto lt = class_lifetimes::build(*prog, aa, axis);
    // K is a view of P: its own tensor lives [pos(K), pos(R)] ; the tensor of P (viewed by K) lives up to R as well
    const auto k = lt.member_interval(prog->get_node("K"), 0);
    ASSERT_TRUE(k.known);
    EXPECT_EQ(k.birth, *axis.position(prog->get_node("K")));
    EXPECT_EQ(k.death, *axis.position(prog->get_node("R")));
    // E is outside the class: from E to its only reader M
    const auto e = lt.member_interval(prog->get_node("E"), 0);
    ASSERT_TRUE(e.known);
    EXPECT_EQ(e.birth, *axis.position(prog->get_node("E")));
    EXPECT_EQ(e.death, *axis.position(prog->get_node("M")));
    EXPECT_FALSE(e.death_inf);
}

// L25b. Static in-place concat: the class is {holder, predecessors}; birth is the FIRST PREDECESSOR (it is written before the holder runs).
TEST(class_lifetime, L25b_concat_class_birth_is_the_first_predecessor) {
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
    prog->get_node("P1").can_share_buffer(false);
    prog->get_node("P2").can_share_buffer(false);
    force_order(*prog, {"in1", "in2", "in3", "P1", "Z", "P2", "C", "Q", "O"});
    auto aa = alias_analysis::build(*prog);
    const auto axis = exec_axis::build(*prog);
    ASSERT_TRUE(axis.valid());
    const auto lt = class_lifetimes::build(*prog, aa, axis);
    const auto* c = aa.class_of(prog->get_node("C"), 0);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c, aa.class_of(prog->get_node("P1"), 0));
    EXPECT_EQ(c, aa.class_of(prog->get_node("P2"), 0));
    const auto& iv = lt.interval_of(*c);
    ASSERT_TRUE(iv.known);
    EXPECT_EQ(iv.birth, *axis.position(prog->get_node("P1")));   // not the position of the holder C
    EXPECT_LT(iv.birth, *axis.position(prog->get_node("C")));
    EXPECT_EQ(iv.death, *axis.position(prog->get_node("Q")));
    // Z sits inside [birth, death] but is not a member: the (class x node) point scan of S4 needs exactly this fact
    const auto pz = *axis.position(prog->get_node("Z"));
    EXPECT_TRUE(iv.birth <= pz && pz <= iv.death);
    EXPECT_TRUE(lt.classes_disjoint(prog->get_node("C"), prog->get_node("Z")));
    // the holder is the only pool allocator (predecessors are zero-alloc members)
    EXPECT_TRUE(has_node(lt.pool_allocators(*c), prog->get_node("C")));
}

// L25c. A class with a user visible output member lives to infinity.
TEST(class_lifetime, L25c_output_class_is_infinite) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(reorder("OUTV", input_info("P"), format::bfyx, data_types::f32));
    auto prog = make_hand(eng, t, {"OUTV"});
    prog->get_node("OUTV").can_be_optimized(true);
    force_order(*prog, {"in", "P", "OUTV"});
    auto aa = alias_analysis::build(*prog);
    const auto axis = exec_axis::build(*prog);
    ASSERT_TRUE(axis.valid());
    const auto lt = class_lifetimes::build(*prog, aa, axis);
    const auto* c = aa.class_of(prog->get_node("P"), 0);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(c, aa.class_of(prog->get_node("OUTV"), 0));
    const auto& iv = lt.interval_of(*c);
    ASSERT_TRUE(iv.known);
    EXPECT_TRUE(iv.death_inf);
    EXPECT_EQ(iv.birth, *axis.position(prog->get_node("P")));
}

// L25d. A class whose member or reader has no position has an unknown interval and is ineligible.
TEST(class_lifetime, L25d_missing_position_makes_the_class_ineligible) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, view_topology(), {"O"});
    prog->get_node("K").can_be_optimized(true);
    force_order(*prog, view_order);
    auto aa = alias_analysis::build(*prog);
    const auto full = exec_axis::build(*prog);
    ASSERT_TRUE(full.valid());
    std::unordered_map<const program_node*, uint32_t> pos;
    for (auto* n : prog->get_processing_order()) {
        if (auto p = full.position(*n)) {
            if (n->id() != "R") {   // R is the reader of the view: dropping it leaves a reader without a position
                pos.emplace(n, *p);
            }
        }
    }
    const auto lt = class_lifetimes::build(*prog, aa, exec_axis::for_test(exec_axis::status::ok, pos));
    const auto* c = aa.class_of(prog->get_node("P"), 0);
    ASSERT_NE(c, nullptr);
    EXPECT_FALSE(lt.interval_of(*c).known);
    const auto m = lt.class_ineligible_mask(*c);
    EXPECT_NE(m & li_unknown_interval, 0u);
    EXPECT_NE(m & li_no_position, 0u);
    EXPECT_FALSE(lt.eligible(prog->get_node("P")));
    EXPECT_FALSE(lt.eligible(prog->get_node("K")));
    // a class of E is untouched (E and M have positions)
    const auto* ce = aa.class_of(prog->get_node("E"), 0);
    ASSERT_NE(ce, nullptr);
    EXPECT_TRUE(lt.interval_of(*ce).known);
}

// L25e. An invalid axis (not normalized / not in order) makes every interval unknown and every node ineligible.
TEST(class_lifetime, L25e_invalid_axis_is_all_ineligible) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, view_topology(), {"O"});
    force_order(*prog, view_order);
    auto aa = alias_analysis::build(*prog);
    const auto lt = class_lifetimes::build(*prog, aa, exec_axis::for_test(exec_axis::status::not_normalized, {}));
    EXPECT_FALSE(lt.valid());
    for (auto* n : prog->get_processing_order()) {
        EXPECT_NE(lt.node_ineligible_mask(*n) & li_no_position, 0u) << n->id();
        EXPECT_FALSE(lt.eligible(*n)) << n->id();
        const auto* c = aa.class_of(*n, 0);
        if (c != nullptr) {
            EXPECT_FALSE(lt.interval_of(*c).known) << n->id();
        }
    }
}

// L25f. Reason bits: a reader that is shape_of / the input layout external root.
TEST(class_lifetime, L25f_reason_bits) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, view_topology(), {"O"});
    force_order(*prog, view_order);
    auto aa = alias_analysis::build(*prog);
    const auto axis = exec_axis::build(*prog);
    ASSERT_TRUE(axis.valid());
    const auto lt = class_lifetimes::build(*prog, aa, axis);
    // input_layout: the buffer belongs to the user (external root)
    EXPECT_NE(lt.node_ineligible_mask(prog->get_node("in")) & li_external, 0u);
    // a plain compute node in a plain class: eligible (no fallback, no persistent, no external, positions known)
    EXPECT_EQ(lt.node_ineligible_mask(prog->get_node("E")), 0u);
    EXPECT_TRUE(lt.eligible(prog->get_node("M")));
}

// L12 (C5 / F27): selective_ssm re-points output 1 to input 5 without an alias edge. The engine marks the output classes and the WHOLE class of
// input 5 (owner + views) as unmodeled. White-box on the mask: no execution needed, so this never depends on the runtime behaviour.
TEST(class_lifetime, L12_selective_ssm_unmodeled_alias_marks_whole_classes) {
    auto& eng = get_test_engine();
    const int64_t heads = 2, groups = 1, head_dim = 4, state_size = 8;
    topology t;
    t.add(input_layout("A", layout{ov::PartialShape{heads}, data_types::f32, format::bfyx}));
    t.add(input_layout("dt", layout{ov::PartialShape{-1, -1, heads}, data_types::f32, format::bfyx}));
    t.add(input_layout("B", layout{ov::PartialShape{-1, -1, groups, state_size}, data_types::f32, format::bfyx}));
    t.add(input_layout("x", layout{ov::PartialShape{-1, -1, heads, head_dim}, data_types::f32, format::bfyx}));
    t.add(input_layout("C", layout{ov::PartialShape{-1, -1, groups, state_size}, data_types::f32, format::bfyx}));
    t.add(input_layout("state_in", layout{ov::PartialShape{-1, heads, head_dim, state_size}, data_types::f32, format::bfyx}));
    t.add(activation("state", input_info("state_in"), activation_func::relu));
    t.add(selective_ssm("selective_ssm", {input_info("A"), input_info("dt"), input_info("B"), input_info("x"), input_info("C"), input_info("state")}));
    t.add(reorder("output", input_info("selective_ssm", 0), format::bfyx, data_types::f32));
    t.add(reorder("state_output", input_info("selective_ssm", 1), format::bfyx, data_types::f32));
    program::ptr prog = make_hand(eng, t, {"output", "state_output"});
    auto aa = alias_analysis::build(*prog);
    // the mask does not need positions: use the real axis if it is valid, an empty valid-looking one otherwise
    const auto axis = exec_axis::build(*prog);
    const auto lt = class_lifetimes::build(*prog, aa, axis);
    auto& ssm = prog->get_node("selective_ssm");
    ASSERT_GE(ssm.get_outputs_count(), 2u);
    for (size_t port = 0; port < ssm.get_outputs_count(); ++port) {
        const auto* c = aa.class_of(ssm, static_cast<int32_t>(port));
        ASSERT_NE(c, nullptr) << "port " << port;
        EXPECT_NE(lt.class_ineligible_mask(*c) & li_unmodeled_alias, 0u) << "selective_ssm output port " << port;
    }
    const auto* cs = aa.class_of(prog->get_node("state"), 0);
    ASSERT_NE(cs, nullptr);
    EXPECT_NE(lt.class_ineligible_mask(*cs) & li_unmodeled_alias, 0u) << "the class of input 5 is unmodeled";
    EXPECT_NE(lt.node_ineligible_mask(ssm) & li_unmodeled_alias, 0u);
    EXPECT_NE(lt.node_ineligible_mask(prog->get_node("state")) & li_unmodeled_alias, 0u);
    // an unrelated class stays unmarked: input "A" (an input_layout class that is not selective_ssm's input 5)
    const auto* ca = aa.class_of(prog->get_node("A"), 0);
    ASSERT_NE(ca, nullptr);
    EXPECT_EQ(lt.class_ineligible_mask(*ca) & li_unmodeled_alias, 0u);
}
