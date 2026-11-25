// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Phase 1a slice 1a-0 (tests only): war_safe verdicts on the target shapes of resolve_inplace_conflicts, on OCL hand graphs with
// hypothetical in-place writers. Design: /home/ceciliapeng/repos/reports/memman/layer1/17_p1a_design.zh-CN.md §4 and Appendix A (C1, C2, N2).
//
// Every case first asserts that the construction really formed (no rejected hypothetical edge, view nodes are can_be_optimized and have a view
// edge, the readers sit on the expected side of the writer on exec_axis); otherwise it is GTEST_SKIPped with the reason. Reader/writer order is
// forced by data dependencies (late reader consumes the writer's output, early reader feeds the writer through another input); the one case
// where that is impossible (assign has no output) uses force_order and asserts the resulting positions.

#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "alias_analysis.h"
#include "alias_validation.h"
#include "exec_axis.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/kv_cache.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/op/interpolate.hpp"
#include "pass_manager.h"
#include "primitive_inst.h"
#include "primitive_type.h"
#include "program_helpers.h"
#include "program_node.h"
#include "program_wrapper.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;
using namespace ::tests;

namespace {

ExecutionConfig make_cfg(engine& eng, const std::vector<std::string>& outputs) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    if (!outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(outputs));
    }
    return config;
}

program::ptr make_hand(engine& eng, const topology& t, const std::vector<std::string>& outs) {
    auto prog = program::build_program(eng, t, make_cfg(eng, outs), /*is_internal=*/false, /*no_optimizations=*/true);
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

layout lay(const ov::PartialShape& s) {
    return layout{s, data_types::f32, format::bfyx};
}

const ov::PartialShape kA{1, 4, 8, 8};

struct war_run {
    alias_analysis aa;
    exec_axis ax;
    war_result res;
};

std::vector<hypothetical_inplace> hyp_of(program& prog, const std::vector<std::pair<std::string, int>>& hyp) {
    std::vector<hypothetical_inplace> extra;
    for (const auto& h : hyp) {
        extra.push_back({&prog.get_node(h.first), h.second});
    }
    return extra;
}

// Builds the engine with the hypothetical edges `hyp` and queries the writer `id` (reading dependency `dep`).
war_run war_of(program& prog, const std::string& id, int dep, const std::vector<std::pair<std::string, int>>& hyp) {
    auto extra = hyp_of(prog, hyp);
    war_run r{alias_analysis::build(prog, compile_time_envelope{}, &extra), exec_axis::build(prog), {}};
    r.res = war_safe(r.aa, r.ax, war_query{&prog.get_node(id), dep, 0});
    return r;
}

std::string why(const war_result& r) {
    std::ostringstream os;
    os << "verdict=" << to_string(r.verdict) << " reasons=0x" << std::hex << r.reasons << std::dec;
    for (const auto& w : r.witnesses) {
        os << " witness(" << (w.reader ? w.reader->id() : "?") << "@" << w.reader_pos << " via " << (w.via_node ? w.via_node->id() : "?") << ")";
    }
    return os.str();
}

int rank_of(war_verdict v) {
    return v == war_verdict::Safe ? 0 : (v == war_verdict::Unknown ? 1 : 2);
}

bool has_view_edge(const alias_analysis& aa, const program_node& member) {
    for (const auto& e : aa.edges()) {
        if (e.origin == edge_origin::view && e.member_node == &member) {
            return true;
        }
    }
    return false;
}

bool witness_has(const war_result& r, const program_node& reader, const program_node* via) {
    for (const auto& w : r.witnesses) {
        if (w.reader == &reader && (via == nullptr || w.via_node == via)) {
            return true;
        }
    }
    return false;
}

#define EXPECT_VERDICT(r, v) EXPECT_EQ((r).res.verdict, v) << why((r).res)
#define EXPECT_REASON(r, bit) EXPECT_NE((r).res.reasons & (bit), 0u) << #bit << " expected; " << why((r).res)
#define EXPECT_NO_REASON(r, bit) EXPECT_EQ((r).res.reasons & (bit), 0u) << #bit << " unexpected; " << why((r).res)

// Construction preconditions shared by all cases: returns false (after GTEST_SKIP-worthy condition) when something is not formed.
#define REQUIRE_FORMED(cond, msg)                                                           \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            GTEST_SKIP() << "construction not formed: " << msg;                             \
        }                                                                                   \
    } while (0)

bool before(const exec_axis& ax, program& p, const std::string& a, const std::string& b) {
    const auto pa = ax.position(p.get_node(a));
    const auto pb = ax.position(p.get_node(b));
    return pa && pb && *pa < *pb;
}

// ---- row 11: the writer overwrites a VIEW V of O ------------------------------------------------------------------------------------
enum class view_kind { reshape_view, resample_view };

// O = relu(in); V = view(O) [cbo]; W = sum(F, V) writes V in place (dep 1).
//  early: the other reader R = abs(O) feeds W through its first input (R runs before W by data dependency);
//  late : Z = prod(O, W') consumes W (through a reshape back for the reshape view), so it runs after W and reads O.
void run_row11(view_kind kind, bool late) {
    auto& eng = get_test_engine();
    const ov::PartialShape S = kind == view_kind::reshape_view ? ov::PartialShape{1, 4, 64, 1} : kA;
    auto view_of = [&](topology& t, const std::string& id, const std::string& src) {
        if (kind == view_kind::reshape_view) {
            t.add(reshape(id, input_info(src), false, std::vector<int64_t>{1, 4, 64, 1}, S));
        } else {
            t.add(resample(id, input_info(src), tensor(1, 4, 8, 8), 1, ov::op::v4::Interpolate::InterpolateMode::NEAREST));
        }
    };
    topology t;
    t.add(input_layout("in", lay(kA)));
    t.add(activation("O", input_info("in"), activation_func::relu));
    view_of(t, "V", "O");
    std::string first = "F";
    if (late) {
        t.add(input_layout("f_in", lay(S)));
        t.add(activation("F", input_info("f_in"), activation_func::relu));
    } else {
        t.add(activation("R", input_info("O"), activation_func::abs));
        if (kind == view_kind::reshape_view) {
            t.add(reshape("Rr", input_info("R"), false, std::vector<int64_t>{1, 4, 64, 1}, S));
            first = "Rr";
        } else {
            first = "R";
        }
    }
    t.add(eltwise("W", {input_info(first), input_info("V")}, eltwise_mode::sum));
    std::vector<std::string> outs;
    if (late) {
        std::string w_out = "W";
        if (kind == view_kind::reshape_view) {
            t.add(reshape("Wb", input_info("W"), false, std::vector<int64_t>{1, 4, 8, 8}, kA));
            w_out = "Wb";
        }
        t.add(eltwise("Z", {input_info("O"), input_info(w_out)}, eltwise_mode::prod));
        outs = {"Z"};
    } else {
        outs = {"W"};
    }
    auto prog = make_hand(eng, t, outs);
    prog->get_node("V").can_be_optimized(true);
    auto r = war_of(*prog, "W", 1, {{"W", 1}});
    REQUIRE_FORMED(r.aa.rejected_hypotheticals().empty(), "a hypothetical edge was rejected (N13)");
    REQUIRE_FORMED(prog->get_node("V").can_be_optimized() && has_view_edge(r.aa, prog->get_node("V")), "V is not a view of O");
    REQUIRE_FORMED(r.ax.valid() && before(r.ax, *prog, "V", "W"), "expected V before W on exec_axis");
    if (late) {
        REQUIRE_FORMED(before(r.ax, *prog, "W", "Z"), "Z must run after W");
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_late_reader);
        ASSERT_FALSE(r.res.witnesses.empty());
        EXPECT_TRUE(witness_has(r.res, prog->get_node("Z"), &prog->get_node("O"))) << "witness must be Z via O: " << why(r.res);
    } else {
        REQUIRE_FORMED(before(r.ax, *prog, "R", "W"), "R must run before W");
        // distinguishing: O has another reader (R), but it is early, so overwriting the view is safe
        EXPECT_VERDICT(r, war_verdict::Safe);
    }
}

}  // namespace

TEST(inplace_conflict_resolution, R1A0_row11_reshape_view_owner_late_reader_is_unsafe) {
    run_row11(view_kind::reshape_view, true);
}

TEST(inplace_conflict_resolution, R1A0_row11_reshape_view_owner_early_reader_is_safe) {
    run_row11(view_kind::reshape_view, false);
}

TEST(inplace_conflict_resolution, R1A0_row11_resample_view_owner_other_reader_early) {
    run_row11(view_kind::resample_view, false);
}

TEST(inplace_conflict_resolution, R1A0_row11_resample_view_owner_other_reader_late) {
    run_row11(view_kind::resample_view, true);
}

// F29 shape (row11b): the writer overwrites an ANCESTOR A of a view R; R's reader Y runs after the writer.
//   A = relu(in); R = resample(A) [cbo]; X = abs(R); Cc = abs(X); C = sum(Cc, A) writes A (dep 1); Y = prod(R, relu(C)).
TEST(inplace_conflict_resolution, R1A0_row11b_ancestor_view_late_reader_is_unsafe_bfyx) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", lay(kA)));
    t.add(activation("A", input_info("in"), activation_func::relu));
    t.add(resample("R", input_info("A"), tensor(1, 4, 8, 8), 1, ov::op::v4::Interpolate::InterpolateMode::NEAREST));
    t.add(activation("X", input_info("R"), activation_func::abs));
    t.add(activation("Cc", input_info("X"), activation_func::abs));
    t.add(eltwise("C", {input_info("Cc"), input_info("A")}, eltwise_mode::sum));
    t.add(activation("fC", input_info("C"), activation_func::relu));
    t.add(eltwise("Y", {input_info("R"), input_info("fC")}, eltwise_mode::prod));
    auto prog = make_hand(eng, t, {"Y"});
    prog->get_node("R").can_be_optimized(true);
    auto r = war_of(*prog, "C", 1, {{"C", 1}});
    REQUIRE_FORMED(r.aa.rejected_hypotheticals().empty(), "a hypothetical edge was rejected (N13)");
    REQUIRE_FORMED(prog->get_node("R").can_be_optimized() && has_view_edge(r.aa, prog->get_node("R")), "R is not a view of A");
    REQUIRE_FORMED(r.ax.valid() && before(r.ax, *prog, "R", "C") && before(r.ax, *prog, "C", "Y"), "expected R < C < Y on exec_axis");
    EXPECT_VERDICT(r, war_verdict::Unsafe);
    EXPECT_REASON(r, wr_late_reader);
    EXPECT_TRUE(witness_has(r.res, prog->get_node("Y"), &prog->get_node("R"))) << "witness must be Y via R: " << why(r.res);
}

// A network-output view of the overwritten tensor makes the writer Unsafe (virtual last reader), even though the writer runs after the view.
//   P = relu(in); V = reshape(P) [cbo, network output]; E = abs(V); Eb = reshape(E); W = sum(P, Eb) writes P (dep 0): W runs after V.
TEST(inplace_conflict_resolution, R1A0_downstream_view_is_network_output_is_unsafe) {
    auto& eng = get_test_engine();
    const ov::PartialShape B{1, 4, 64, 1};
    topology t;
    t.add(input_layout("in", lay(kA)));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(reshape("V", input_info("P"), false, std::vector<int64_t>{1, 4, 64, 1}, B));
    t.add(activation("E", input_info("V"), activation_func::abs));
    t.add(reshape("Eb", input_info("E"), false, std::vector<int64_t>{1, 4, 8, 8}, kA));
    t.add(eltwise("W", {input_info("P"), input_info("Eb")}, eltwise_mode::sum));
    auto prog = make_hand(eng, t, {"V", "W"});
    prog->get_node("V").can_be_optimized(true);
    auto r = war_of(*prog, "W", 0, {{"W", 0}});
    REQUIRE_FORMED(r.aa.rejected_hypotheticals().empty(), "a hypothetical edge was rejected (N13)");
    REQUIRE_FORMED(prog->get_node("V").can_be_optimized() && prog->get_node("V").is_output() && has_view_edge(r.aa, prog->get_node("V")),
                   "V is not a cbo network-output view of P");
    REQUIRE_FORMED(r.ax.valid() && before(r.ax, *prog, "V", "W"), "W must run after V");
    EXPECT_VERDICT(r, war_verdict::Unsafe);
    EXPECT_REASON(r, wr_visible_version);
}

// Y3 lock (current behaviour, NOT the desired one): the assign's version set starts at the aliases of the variable (read_value outputs); the
// in-place output of a kv_cache of the same variable is not among them, so its readers never appear as witnesses. Slice 1a-3 fixes Y3 and
// must FLIP the last assertion.
//   rv = read_value(v0); kv = kv_cache(rv, new_kv, v0); X = abs(kv); as = assign(X, v0) [cpu]; Lk = abs(kv) forced after as.
TEST(inplace_conflict_resolution, R1A0_y3_lock_assign_witness_excludes_kv_reader) {
    auto& eng = get_test_engine();
    const ov::PartialShape state_shape{1, 2, 4, 8};
    const layout kv_out{ov::PartialShape{1, 2, 5, 8}, data_types::f16, format::bfyx};
    topology t;
    t.add(input_layout("state", layout{state_shape, data_types::f16, format::bfyx}));
    t.add(read_value{"rv", {input_info("state")}, "v0", {layout{state_shape, data_types::f16, format::bfyx}}});
    t.add(input_layout("new_kv", layout{ov::PartialShape{1, 2, 1, 8}, data_types::f16, format::bfyx}));
    t.add(kv_cache("kv", {input_info("rv"), input_info("new_kv")}, ov::op::util::VariableInfo{state_shape, ov::element::f16, "v0"}, 2, 0, false, false,
                   false));
    t.add(activation("X", input_info("kv"), activation_func::abs));
    t.add(assign("as", {input_info("X")}, "v0", kv_out));
    t.add(activation("Lk", input_info("kv"), activation_func::abs));
    auto prog = make_hand(eng, t, {"as", "Lk"});
    prog->get_node("as").set_preferred_impl_type(impl_types::cpu);   // assign only has a CPU implementation
    force_order(*prog, {"rv", "kv", "X", "as", "Lk"});
    auto aa = alias_analysis::build(*prog);
    const auto ax = exec_axis::build(*prog);
    REQUIRE_FORMED(ax.valid() && before(ax, *prog, "as", "Lk") && before(ax, *prog, "kv", "as"), "expected kv < as < Lk on exec_axis");
    const war_query* q = nullptr;
    auto writers = enumerate_inplace_writers(aa);
    for (const auto& w : writers) {
        if (w.writer == &prog->get_node("as")) {
            q = &w;
        }
    }
    REQUIRE_FORMED(q != nullptr, "assign is not an enumerated in-place writer");
    auto r = war_safe(aa, ax, *q);
    EXPECT_EQ(r.verdict, war_verdict::Unknown) << why(r);
    EXPECT_NE(r.reasons & wr_cpu_writer, 0u) << why(r);
    for (const auto& w : r.witnesses) {
        EXPECT_NE(w.reader, &prog->get_node("Lk")) << "Y3 current behaviour: a reader of the kv_cache output is NOT a witness of the assign (flip in 1a-3)";
        EXPECT_NE(w.reader, &prog->get_node("X")) << why(r);
    }
}

// ---- C2 (audit Appendix A): executable Lemma 1 and the downward in-place edge (N2) ----------------------------------------------------

// C2(a). Two hypothetical writers on one chain: W1 overwrites P (late reader Z1 of P => Unsafe); W2 overwrites W1's output (Safe).
// W1 is downgraded; the lemma says W2 stays Safe on the engine built with W2 alone.
//   P = relu(in); W1 = sum(P, f1) [P]; W2 = sum(W1, f2) [W1]; Z1 = prod(P, W2) runs after W2, hence after W1.
TEST(inplace_conflict_resolution, R1A0_c2a_second_writer_verdict_does_not_get_worse_after_first_is_downgraded) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", lay(kA)));
    t.add(input_layout("f1", lay(kA)));
    t.add(input_layout("f2", lay(kA)));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(eltwise("W1", {input_info("P"), input_info("f1")}, eltwise_mode::sum));
    t.add(eltwise("W2", {input_info("W1"), input_info("f2")}, eltwise_mode::sum));
    t.add(eltwise("Z1", {input_info("P"), input_info("W2")}, eltwise_mode::prod));
    auto prog = make_hand(eng, t, {"Z1"});

    auto both = war_of(*prog, "W1", 0, {{"W1", 0}, {"W2", 0}});
    auto both_w2 = war_safe(both.aa, both.ax, war_query{&prog->get_node("W2"), 0, 0});
    REQUIRE_FORMED(both.aa.rejected_hypotheticals().empty(), "a hypothetical edge was rejected (N13)");
    REQUIRE_FORMED(both.ax.valid() && before(both.ax, *prog, "W1", "W2") && before(both.ax, *prog, "W2", "Z1"), "expected W1 < W2 < Z1");
    EXPECT_VERDICT(both, war_verdict::Unsafe);
    EXPECT_REASON(both, wr_late_reader);
    EXPECT_EQ(both_w2.verdict, war_verdict::Safe) << "W2 with both writers: " << why(both_w2);

    // W1 is downgraded: rebuild the engine with W2 only
    auto only = war_of(*prog, "W2", 0, {{"W2", 0}});
    REQUIRE_FORMED(only.aa.rejected_hypotheticals().empty(), "W2-only hypothetical edge was rejected (N13)");
    EXPECT_LE(rank_of(only.res.verdict), rank_of(both_w2.verdict)) << "Lemma 1 violated: W2 got worse after W1 was downgraded: " << why(only.res);
    EXPECT_VERDICT(only, war_verdict::Safe);
}

// C2(c) / N2. Downward direction: Wd reads P in place BEFORE W2 (so W2 does not see Od); Z reads Od AFTER W2.
//   P = relu(in); Wd = sum(P, fd) [P]  (Od = Wd's output aliases P); Ad = abs(Od) feeds W2 (forces Wd < W2);
//   W2 = sum(P, Ad) [P]; Z = prod(Od, W2) runs after W2.
// Wd must not be Safe (W2, a reader of P, runs after it); W2 is Safe on the version set of P; with Wd downgraded W2 stays Safe.
TEST(inplace_conflict_resolution, R1A0_c2c_downward_inplace_edge_the_earlier_writer_is_the_one_that_must_be_downgraded) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", lay(kA)));
    t.add(input_layout("fd", lay(kA)));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(eltwise("Wd", {input_info("P"), input_info("fd")}, eltwise_mode::sum));
    t.add(activation("Ad", input_info("Wd"), activation_func::abs));
    t.add(eltwise("W2", {input_info("P"), input_info("Ad")}, eltwise_mode::sum));
    t.add(eltwise("Z", {input_info("Wd"), input_info("W2")}, eltwise_mode::prod));
    auto prog = make_hand(eng, t, {"Z"});

    auto rd = war_of(*prog, "Wd", 0, {{"Wd", 0}, {"W2", 0}});
    auto r2 = war_safe(rd.aa, rd.ax, war_query{&prog->get_node("W2"), 0, 0});
    REQUIRE_FORMED(rd.aa.rejected_hypotheticals().empty(), "a hypothetical edge was rejected (N13)");
    REQUIRE_FORMED(rd.ax.valid() && before(rd.ax, *prog, "Wd", "Ad") && before(rd.ax, *prog, "Ad", "W2") && before(rd.ax, *prog, "W2", "Z"),
                   "expected Wd < Ad < W2 < Z");
    EXPECT_NE(rd.res.verdict, war_verdict::Safe) << "Wd must be downgrade-worthy: " << why(rd.res);
    EXPECT_VERDICT(rd, war_verdict::Unsafe);
    EXPECT_REASON(rd, wr_late_reader);
    EXPECT_TRUE(witness_has(rd.res, prog->get_node("W2"), nullptr)) << "witness must be W2 (a reader of P after Wd): " << why(rd.res);
    EXPECT_EQ(r2.verdict, war_verdict::Safe) << "W2 is kept: " << why(r2);

    auto only = war_of(*prog, "W2", 0, {{"W2", 0}});
    REQUIRE_FORMED(only.aa.rejected_hypotheticals().empty(), "W2-only hypothetical edge was rejected (N13)");
    EXPECT_VERDICT(only, war_verdict::Safe);
}

// =====================================================================================================================================
// Slice 1a-1: the planner (pure function over war_safe) on OCL hand graphs with hypothetical writers. Design 17 §5.2, tests P1..P6.
// =====================================================================================================================================
namespace {

// P = relu(in); Y = abs(P) (reader of P); W = sum(P, f) writes P (hypothetical, dep 0); Z = prod(Y, W).
topology plan_fan(bool with_shape_of = false) {
    topology t;
    t.add(input_layout("in", lay(kA)));
    t.add(input_layout("f_in", lay(kA)));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(activation("Y", input_info("P"), activation_func::abs));
    t.add(eltwise("W", {input_info("P"), input_info("f_in")}, eltwise_mode::sum));
    t.add(eltwise("Z", {input_info("Y"), input_info("W")}, eltwise_mode::prod));
    if (with_shape_of) {
        t.add(shape_of("S", input_info("P"), data_types::i32));
    }
    return t;
}

struct plan_run {
    alias_analysis aa;
    exec_axis ax;
    inplace_conflict_plan plan;
};

plan_run plan_of(program& prog, const std::vector<std::pair<std::string, int>>& hyp) {
    auto extra = hyp_of(prog, hyp);
    plan_run r{alias_analysis::build(prog, compile_time_envelope{}, &extra), exec_axis::build(prog), {}};
    r.plan = plan_inplace_conflict_resolution(r.aa, r.ax, /*include_hypothetical=*/true);
    return r;
}

std::string plan_str(const inplace_conflict_plan& plan) {
    std::ostringstream os;
    for (const auto& a : plan.actions) {
        os << " [" << a.writer->id() << " dep " << a.mem_dep << " " << to_string(a.origin) << " why=" << to_string(a.why) << " reasons=0x" << std::hex
           << a.r.reasons << std::dec << "]";
    }
    return os.str();
}

}  // namespace

// P1: a hypothetical Safe writer is kept.
TEST(inplace_conflict_resolution, P1_planner_safe_writer_is_kept) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, plan_fan(), {"Z"});
    force_order(*prog, {"P", "Y", "W", "Z"});
    auto r = plan_of(*prog, {{"W", 0}});
    REQUIRE_FORMED(r.aa.rejected_hypotheticals().empty(), "a hypothetical edge was rejected (N13)");
    REQUIRE_FORMED(r.ax.valid() && before(r.ax, *prog, "Y", "W"), "Y must run before W");
    ASSERT_EQ(r.plan.actions.size(), 1u) << plan_str(r.plan);
    EXPECT_EQ(r.plan.actions[0].why, war_downgrade_reason::none) << plan_str(r.plan);
    EXPECT_EQ(r.plan.actions[0].origin, edge_origin::hypothetical);
    EXPECT_EQ(r.plan.actions[0].r.verdict, war_verdict::Safe);
}

// P2: a late reader of the overwritten tensor makes the writer a downgrade (unsafe, late_reader).
TEST(inplace_conflict_resolution, P2_planner_late_reader_is_unsafe) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, plan_fan(), {"Z"});
    force_order(*prog, {"P", "W", "Y", "Z"});
    auto r = plan_of(*prog, {{"W", 0}});
    REQUIRE_FORMED(r.aa.rejected_hypotheticals().empty(), "a hypothetical edge was rejected (N13)");
    REQUIRE_FORMED(r.ax.valid() && before(r.ax, *prog, "W", "Y"), "W must run before Y");
    ASSERT_EQ(r.plan.actions.size(), 1u) << plan_str(r.plan);
    EXPECT_EQ(r.plan.actions[0].why, war_downgrade_reason::unsafe) << plan_str(r.plan);
    EXPECT_NE(r.plan.actions[0].r.reasons & wr_late_reader, 0u) << plan_str(r.plan);
}

// P3: a reader that may run on the CPU (rigid exclusion) gives a node-level Unknown, which is downgraded.
TEST(inplace_conflict_resolution, P3_planner_cpu_reader_is_node_unknown) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, plan_fan(), {"Z"});
    prog->get_node("Y").set_preferred_impl_type(impl_types::cpu);
    force_order(*prog, {"P", "Y", "W", "Z"});
    auto r = plan_of(*prog, {{"W", 0}});
    REQUIRE_FORMED(r.aa.rejected_hypotheticals().empty(), "a hypothetical edge was rejected (N13)");
    REQUIRE_FORMED(r.ax.valid() && before(r.ax, *prog, "Y", "W"), "Y must run before W");
    ASSERT_EQ(r.plan.actions.size(), 1u) << plan_str(r.plan);
    EXPECT_EQ(r.plan.actions[0].why, war_downgrade_reason::node_unknown) << plan_str(r.plan);
    EXPECT_NE(r.plan.actions[0].r.reasons & wr_cpu_reader, 0u) << plan_str(r.plan);
}

// P4: a shape_of reader is ignored (keep); a reader inside a shape_of subgraph is a node-level Unknown.
TEST(inplace_conflict_resolution, P4_planner_shape_of_reader_kept_and_shape_of_subgraph_reader_unknown) {
    auto& eng = get_test_engine();
    {
        auto prog = make_hand(eng, plan_fan(true), {"Z", "S"});
        force_order(*prog, {"P", "Y", "W", "Z"});
        auto r = plan_of(*prog, {{"W", 0}});
        REQUIRE_FORMED(r.aa.rejected_hypotheticals().empty(), "a hypothetical edge was rejected (N13)");
        REQUIRE_FORMED(r.ax.valid() && before(r.ax, *prog, "Y", "W"), "Y must run before W");
        ASSERT_EQ(r.plan.actions.size(), 1u) << plan_str(r.plan);
        EXPECT_EQ(r.plan.actions[0].why, war_downgrade_reason::none) << "the shape_of reader must not count: " << plan_str(r.plan);
    }
    {
        auto prog = make_hand(eng, plan_fan(), {"Z"});
        prog->get_node("Y").set_in_shape_of_subgraph(true);
        force_order(*prog, {"P", "Y", "W", "Z"});
        auto r = plan_of(*prog, {{"W", 0}});
        REQUIRE_FORMED(r.aa.rejected_hypotheticals().empty(), "a hypothetical edge was rejected (N13)");
        ASSERT_EQ(r.plan.actions.size(), 1u) << plan_str(r.plan);
        EXPECT_EQ(r.plan.actions[0].why, war_downgrade_reason::node_unknown) << plan_str(r.plan);
        EXPECT_NE(r.plan.actions[0].r.reasons & wr_shape_of_subgraph_reader, 0u) << plan_str(r.plan);
    }
}

// P5: the plan order is (exec position, uid, dep) whatever the order of the hypothetical list; a dynamic writer is not planned (W17 lock).
TEST(inplace_conflict_resolution, P5_planner_order_is_deterministic_and_dynamic_writers_are_excluded) {
    auto& eng = get_test_engine();
    {
        topology t;
        t.add(input_layout("in", lay(kA)));
        t.add(input_layout("f1", lay(kA)));
        t.add(input_layout("f2", lay(kA)));
        t.add(activation("P", input_info("in"), activation_func::relu));
        t.add(eltwise("W1", {input_info("P"), input_info("f1")}, eltwise_mode::sum));
        t.add(eltwise("W2", {input_info("W1"), input_info("f2")}, eltwise_mode::sum));
        t.add(activation("Z", input_info("W2"), activation_func::abs));
        auto prog = make_hand(eng, t, {"Z"});
        auto fwd = plan_of(*prog, {{"W1", 0}, {"W2", 0}});
        auto rev = plan_of(*prog, {{"W2", 0}, {"W1", 0}});
        REQUIRE_FORMED(fwd.aa.rejected_hypotheticals().empty() && rev.aa.rejected_hypotheticals().empty(), "a hypothetical edge was rejected (N13)");
        REQUIRE_FORMED(fwd.ax.valid() && before(fwd.ax, *prog, "W1", "W2"), "W1 must run before W2");
        ASSERT_EQ(fwd.plan.actions.size(), 2u) << plan_str(fwd.plan);
        ASSERT_EQ(rev.plan.actions.size(), 2u) << plan_str(rev.plan);
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_EQ(fwd.plan.actions[i].writer, rev.plan.actions[i].writer) << "position " << i;
        }
        EXPECT_EQ(fwd.plan.actions[0].writer, &prog->get_node("W1"));
        EXPECT_EQ(fwd.plan.actions[1].writer, &prog->get_node("W2"));
    }
    {
        topology t;
        t.add(input_layout("in", layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}));
        t.add(activation("P", input_info("in"), activation_func::relu));
        t.add(activation("W", input_info("P"), activation_func::abs));
        auto prog = make_hand(eng, t, {"W"});
        REQUIRE_FORMED(prog->get_node("W").is_dynamic(), "W is not dynamic");
        auto r = plan_of(*prog, {{"W", 0}});
        REQUIRE_FORMED(!enumerate_inplace_writers(r.aa).empty(), "the engine has no in-place writer (vacuity)");
        EXPECT_TRUE(r.plan.actions.empty()) << "a dynamic writer must not be planned: " << plan_str(r.plan);
    }
}

// P6 (D6 distinguishing power): find_unsafe_surviving_inplace_writers reports an Unsafe hypothetical writer when asked to include it, is
// silent for a Safe one, and (as in the production call) ignores hypothetical edges by default.
TEST(inplace_conflict_resolution, P6_find_unsafe_surviving_reports_unsafe_hypothetical_writer) {
    auto& eng = get_test_engine();
    auto late = make_hand(eng, plan_fan(), {"Z"});
    force_order(*late, {"P", "W", "Y", "Z"});
    auto extra = hyp_of(*late, {{"W", 0}});
    auto aa = alias_analysis::build(*late, compile_time_envelope{}, &extra);
    const auto ax = exec_axis::build(*late);
    REQUIRE_FORMED(aa.rejected_hypotheticals().empty() && ax.valid() && before(ax, *late, "W", "Y"), "late construction not formed");
    auto bad = find_unsafe_surviving_inplace_writers(aa, ax, /*include_hypothetical=*/true);
    ASSERT_EQ(bad.size(), 1u);
    EXPECT_EQ(bad[0].writer, &late->get_node("W"));
    EXPECT_EQ(bad[0].why, war_downgrade_reason::unsafe);
    EXPECT_TRUE(find_unsafe_surviving_inplace_writers(aa, ax, false).empty()) << "production D6 must not look at hypothetical edges";

    auto early = make_hand(eng, plan_fan(), {"Z"});
    force_order(*early, {"P", "Y", "W", "Z"});
    auto extra2 = hyp_of(*early, {{"W", 0}});
    auto aa2 = alias_analysis::build(*early, compile_time_envelope{}, &extra2);
    const auto ax2 = exec_axis::build(*early);
    REQUIRE_FORMED(aa2.rejected_hypotheticals().empty() && ax2.valid() && before(ax2, *early, "Y", "W"), "early construction not formed");
    EXPECT_TRUE(find_unsafe_surviving_inplace_writers(aa2, ax2, true).empty());
}

// =====================================================================================================================================
// Slice 1a-1: the executor (resolve_inplace_conflicts inside refreeze_onednn_inplace_post_ops) on real oneDNN sums.
// Every case first asserts that the sum(s) really formed at the freeze (frozen entry is a sum AND the live predicate says sum), otherwise
// GTEST_SKIP with the reason (a SKIP counts as "not covered"). Reader / writer order is forced by data dependencies.
// =====================================================================================================================================
#ifdef ENABLE_ONEDNN_FOR_GPU

#include "intel_gpu/graph/serialization/binary_buffer.hpp"

namespace {

using fusing = onednn_eltwise_fusing_helpers;
const ov::PartialShape kFc{32, 16};

#define P1A_SKIP_IF_NO_IMMAD()                                                                       \
    do {                                                                                             \
        if (!get_test_engine().get_device_info().supports_immad) {                                   \
            GTEST_SKIP() << "the oneDNN in-place sum needs a oneDNN device (supports_immad)";        \
        }                                                                                            \
    } while (0)

std::vector<float> to_f32(memory::ptr mem) {
    std::vector<float> r;
    const size_t count = mem->get_layout().count();
    if (mem->get_layout().data_type == data_types::f16) {
        mem_lock<ov::float16, mem_lock_type::read> lock(mem, get_test_stream());
        for (size_t i = 0; i < count && i < lock.size(); i++) {
            r.push_back(static_cast<float>(lock[i]));
        }
    } else {
        mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
        for (size_t i = 0; i < count && i < lock.size(); i++) {
            r.push_back(lock[i]);
        }
    }
    return r;
}

memory::ptr make_f16(engine& eng, const ov::PartialShape& shape, int variant) {
    layout l{shape, data_types::f16, format::bfyx};
    auto mem = eng.allocate_memory(l);
    std::vector<ov::float16> v(l.count());
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = ov::float16(static_cast<float>(((i * (3 + variant)) % 13)) * 0.0625f + 0.125f * variant);
    }
    set_values(mem, v);
    return mem;
}

memory::ptr make_fc_weights(engine& eng, int64_t o, int64_t i, int variant) {
    layout l{ov::PartialShape{o, i}, data_types::f16, format::bfyx};
    auto mem = eng.allocate_memory(l);
    std::vector<ov::float16> v(l.count());
    for (size_t k = 0; k < v.size(); k++) {
        v[k] = ov::float16(static_cast<float>((k * (5 + variant)) % 7) * 0.03125f - 0.0625f);
    }
    set_values(mem, v);
    return mem;
}

ov::intel_gpu::ImplementationDesc impl_desc(impl_types t) {
    return ov::intel_gpu::ImplementationDesc{format::bfyx, "", t};
}

struct fc_graph {
    topology t;
    ov::intel_gpu::ImplForcingMap forced;
    std::vector<std::string> outs;
    std::vector<std::pair<std::string, layout>> inputs;
    std::vector<std::string> writers;   // nodes expected to hold a formed sum at the freeze
};

ExecutionConfig opt_cfg(engine& eng, const fc_graph& g, bool resolution_on, QueueTypes q = QueueTypes::in_order) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(g.forced));
    config.set_property(ov::intel_gpu::queue_type(q));
    config.set_property(ov::intel_gpu::enable_inplace_conflict_resolution(resolution_on));
    return config;
}

using hook_t = std::function<void(program&)>;

// Same stages as program::build_program(), with a hook between the oneDNN attribute freeze + impl selection and post_optimize_graph.
program::ptr build_staged(engine& eng, const fc_graph& g, bool resolution_on, const hook_t& hook = {}) {
    auto prog = program::build_program(eng, g.t, opt_cfg(eng, g, resolution_on), /*is_internal=*/false, /*no_optimizations=*/true);
    program_wrapper::pre_optimize_graph(*prog, false);
    program_wrapper::run_graph_compilation(*prog);
    if (hook) {
        hook(*prog);
    }
    program_wrapper::post_optimize_graph(*prog, false);
    program_wrapper::prepare_memory_dependencies(*prog);
    program_wrapper::apply_opt_pass<build_implementations>(*prog);
    if (eng.get_device_info().has_separate_cache) {
        program_wrapper::transfer_memory_to_device(*prog);
    }
    return prog;
}

using result_map = std::map<std::string, std::vector<float>>;

result_map run_net(network& net, engine& eng, const fc_graph& g) {
    int variant = 1;
    for (const auto& in : g.inputs) {
        net.set_input_data(in.first, make_f16(eng, in.second.get_partial_shape(), variant++));
    }
    auto outs = net.execute();
    result_map r;
    for (const auto& id : g.outs) {
        r[id] = to_f32(outs.at(id).get_memory());
    }
    return r;
}

result_map run_reference(engine& eng, const fc_graph& g) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network net_ref(eng, g.t, config);
    return run_net(net_ref, eng, g);
}

// Number of elements beyond the fp16 tolerance (abs 2e-2 + 1% relative).
size_t mismatches(const result_map& got, const result_map& ref) {
    size_t bad = 0;
    for (const auto& kv : ref) {
        const auto it = got.find(kv.first);
        if (it == got.end() || it->second.size() != kv.second.size()) {
            return std::numeric_limits<size_t>::max();
        }
        for (size_t i = 0; i < kv.second.size(); i++) {
            if (!(std::fabs(it->second[i] - kv.second[i]) <= 0.02f + 0.01f * std::fabs(kv.second[i]))) {
                bad++;
            }
        }
    }
    return bad;
}

// Dependency index of the formed in-place sum of `w`: the frozen entry is a sum AND the live predicate says sum; -1 otherwise.
int32_t sum_dep_if_formed(const program_node& w) {
    for (const auto& fd : w.get_fused_primitives()) {
        if (fd.is_type<eltwise>() && fd.typed_desc<eltwise>()->mode == eltwise_mode::sum && fd.deps.size() == 1 && fd.has_outer_dep()) {
            const int32_t dep = fd.outer_dep_start_idx;
            if (fusing::is_frozen_inplace_post_op(w, dep, eltwise_mode::sum) && fusing::get_add_fusing_type(w, fd) == fusing::add_fusing_type::sum) {
                return dep;
            }
            return -1;
        }
    }
    return -1;
}

// outer dependency index of the fused sum-mode eltwise of `id` (independent of the frozen state, so also valid after a downgrade)
int32_t outer_dep(program& prog, const std::string& id) {
    for (const auto& fd : prog.get_node(id).get_fused_primitives()) {
        if (fd.is_type<eltwise>() && fd.typed_desc<eltwise>()->mode == eltwise_mode::sum && fd.deps.size() == 1 && fd.has_outer_dep()) {
            return fd.outer_dep_start_idx;
        }
    }
    return -1;
}

bool impl_has_sum_at(const program_node& n, int32_t dep) {
    const auto* impl = n.get_selected_impl();
    if (!impl || !impl->is_onednn()) {
        return false;
    }
    const auto slots = get_compiled_post_op_slots(n.get_fused_primitives_onednn(), static_cast<size_t>(dep),
                                                  {onednn_post_op_type::sum, onednn_post_op_type::optimized_sum});
    return slots.slot && impl->is_sum_post_op(slots.kinds, *slots.slot);
}

hook_t formation_hook(const fc_graph& g, bool& formed, std::string& why_not) {
    return [&g, &formed, &why_not](program& p) {
        for (const auto& w : g.writers) {
            if (!p.has_node(w) || sum_dep_if_formed(p.get_node(w)) < 0) {
                formed = false;
                why_not += " " + w;
                if (p.has_node(w)) {
                    // diagnostics: what the node holds instead of a formed sum
                    for (const auto& fd : p.get_node(w).get_fused_primitives()) {
                        why_not += "{fused " + fd.desc->type_string() + " outer_dep=" + std::to_string(fd.outer_dep_start_idx);
                        if (fd.is_type<eltwise>() && fd.has_outer_dep()) {
                            why_not += " add_fusing_type=" + std::to_string(static_cast<int>(fusing::get_add_fusing_type(p.get_node(w), fd))) +
                                       " frozen_sum=" + std::to_string(fusing::is_frozen_inplace_post_op(p.get_node(w), fd.outer_dep_start_idx, eltwise_mode::sum));
                        }
                        why_not += "}";
                    }
                    why_not += "{frozen_list=" + std::to_string(p.get_node(w).get_fused_primitives_onednn().size()) + "}";
                }
                why_not += "\n   graph:";
                for (auto* n : p.get_processing_order()) {
                    why_not += " " + n->id() + "(" + n->get_primitive()->type_string();
                    for (const auto& fd : n->get_fused_primitives()) {
                        why_not += "+" + fd.desc->type_string() + "[outer=" + std::to_string(fd.outer_dep_start_idx) + "]";
                    }
                    why_not += ")";
                }
            }
        }
    };
}

// the three-way state of a downgraded writer: frozen entry, impl slot and get_reused_eltwmem_idx agree that there is no sum any more
void expect_downgraded(program& prog, const std::string& w_id, int32_t dep) {
    auto& w = prog.get_node(w_id);
    EXPECT_FALSE(fusing::is_frozen_inplace_post_op(w, dep, eltwise_mode::sum)) << w_id << ": frozen entry";
    EXPECT_FALSE(impl_has_sum_at(w, dep)) << w_id << ": impl slot";
    EXPECT_NE(fusing::get_reused_eltwmem_idx(w), dep) << w_id << ": get_reused_eltwmem_idx";
}

void expect_kept(program& prog, const std::string& w_id, int32_t dep) {
    auto& w = prog.get_node(w_id);
    EXPECT_TRUE(fusing::is_frozen_inplace_post_op(w, dep, eltwise_mode::sum)) << w_id << ": frozen entry";
    EXPECT_TRUE(impl_has_sum_at(w, dep)) << w_id << ": impl slot";
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(w), dep) << w_id << ": get_reused_eltwmem_idx";
}

// ---- graphs -------------------------------------------------------------------------------------------------------------------------

layout f16_lay() {
    return layout{kFc, data_types::f16, format::bfyx};
}

void add_fc(fc_graph& g, engine& eng, const std::string& id, const std::string& in, int variant, impl_types t) {
    g.t.add(data(id + "_w", make_fc_weights(eng, 16, 16, variant)));
    g.t.add(fully_connected(id, input_info(in), id + "_w"));
    g.forced[id] = impl_desc(t);
}

// X = FC_ocl(in3); W = FC_onednn(in1); S = sum(W, X); A = relu(S); Z = FC_ocl(A); out_Z.
fc_graph base_graph(engine& eng) {
    fc_graph g;
    g.t.add(input_layout("in1", f16_lay()));
    g.t.add(input_layout("in3", f16_lay()));
    g.inputs = {{"in1", f16_lay()}, {"in3", f16_lay()}};
    add_fc(g, eng, "X", "in3", 1, impl_types::ocl);
    add_fc(g, eng, "W", "in1", 0, impl_types::onednn);
    g.t.add(eltwise("S", {input_info("W"), input_info("X")}, eltwise_mode::sum));
    g.t.add(activation("A", input_info("S"), activation_func::relu));
    add_fc(g, eng, "Z", "A", 2, impl_types::ocl);
    g.t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.outs = {"out_Z"};
    g.writers = {"W"};
    return g;
}

// Residual chain of 3 dep(0) hops (within the walk limit): x_proj -> f1 -> f2 -> p ; res = sum(p, x_proj).
fc_graph chain_graph(engine& eng) {
    fc_graph g;
    g.t.add(input_layout("in", f16_lay()));
    g.inputs = {{"in", f16_lay()}};
    add_fc(g, eng, "x_proj", "in", 1, impl_types::ocl);
    add_fc(g, eng, "f1", "x_proj", 2, impl_types::ocl);
    add_fc(g, eng, "f2", "f1", 3, impl_types::ocl);
    add_fc(g, eng, "p", "f2", 0, impl_types::onednn);
    g.t.add(eltwise("res", {input_info("p"), input_info("x_proj")}, eltwise_mode::sum));
    add_fc(g, eng, "Z", "res", 9, impl_types::ocl);
    g.t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.outs = {"out_Z"};
    g.writers = {"p"};
    return g;
}

// Row 11 on FC: the writer W overwrites V = no-op strided_slice view of O.
//   late : Zp = prod(O, S) reads O after the sum (it consumes the writer's output): the sum must be downgraded.
//   early: R = abs(O) feeds W through its input, so it runs before W: the sum stays.
fc_graph row11_fc_graph(engine& eng, bool late) {
    fc_graph g;
    g.t.add(input_layout("in3", f16_lay()));
    g.inputs = {{"in3", f16_lay()}};
    add_fc(g, eng, "O", "in3", 1, impl_types::ocl);
    g.t.add(strided_slice("V", input_info("O"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
    if (late) {
        g.t.add(input_layout("in1", f16_lay()));
        g.inputs.push_back({"in1", f16_lay()});
        add_fc(g, eng, "W", "in1", 0, impl_types::onednn);
    } else {
        g.t.add(activation("R", input_info("O"), activation_func::abs));
        add_fc(g, eng, "W", "R", 0, impl_types::onednn);
    }
    g.t.add(eltwise("S", {input_info("W"), input_info("V")}, eltwise_mode::sum));
    std::string z_in = "S";
    if (late) {
        // S gets a second user (Zb) so that Zp is not fused into the writer as a post-op (then O would be read by the writer itself)
        g.t.add(eltwise("Zp", {input_info("O"), input_info("S")}, eltwise_mode::prod));
        add_fc(g, eng, "Zb", "S", 5, impl_types::ocl);
        g.t.add(reorder("out_Zb", input_info("Zb"), format::bfyx, data_types::f32));
        g.outs.push_back("out_Zb");
        z_in = "Zp";
    }
    add_fc(g, eng, "Z", z_in, 2, impl_types::ocl);
    g.t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.outs.push_back("out_Z");
    g.writers = {"W"};
    return g;
}

// Row 11b on FC (F29): the writer Cc overwrites A, a view R of A has another reader Y that runs after the sum.
//   A = FC(in); R = strided_slice(A) [view]; X = FC(R); Cc = FC_onednn(X); C = sum(Cc, A); Y = prod(R, relu(C)).
fc_graph row11b_fc_graph(engine& eng) {
    fc_graph g;
    g.t.add(input_layout("in", f16_lay()));
    g.inputs = {{"in", f16_lay()}};
    add_fc(g, eng, "A", "in", 1, impl_types::ocl);
    g.t.add(strided_slice("R", input_info("A"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
    add_fc(g, eng, "X", "R", 2, impl_types::ocl);
    add_fc(g, eng, "Cc", "X", 0, impl_types::onednn);
    g.t.add(eltwise("C", {input_info("Cc"), input_info("A")}, eltwise_mode::sum));
    g.t.add(activation("fC", input_info("C"), activation_func::relu));
    g.t.add(eltwise("Y", {input_info("R"), input_info("fC")}, eltwise_mode::prod));
    g.t.add(reorder("out_Y", input_info("Y"), format::bfyx, data_types::f32));
    g.t.add(reorder("out_C", input_info("C"), format::bfyx, data_types::f32));   // second user of C: the activation / prod are not fused into Cc
    g.outs = {"out_Y", "out_C"};
    g.writers = {"Cc"};
    return g;
}

// C2b: a chain {O-owned root -> W1 -> W2} whose MIDDLE link W1 is downgraded (late reader Zp of O) while W2 (writes W1's output) stays.
//   O = FC; V = strided_slice(O) [view]; W1 = FC_onednn(in1); S1 = sum(W1, V); a1 = abs(S1); W2 = FC_onednn(a1); S2 = sum(W2, S1) (a residual
//   on S1: direct ancestor path W2 -> a1 -> S1, and W2 is the latest of the two so it hosts the sum); Zp = prod(O, S2) (S2 has a second user Zb).
fc_graph chain_mid_graph(engine& eng) {
    fc_graph g;
    g.t.add(input_layout("in3", f16_lay()));
    g.t.add(input_layout("in1", f16_lay()));
    g.inputs = {{"in3", f16_lay()}, {"in1", f16_lay()}};
    add_fc(g, eng, "O", "in3", 1, impl_types::ocl);
    g.t.add(strided_slice("V", input_info("O"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
    add_fc(g, eng, "W1", "in1", 0, impl_types::onednn);
    g.t.add(eltwise("S1", {input_info("W1"), input_info("V")}, eltwise_mode::sum));
    g.t.add(activation("a1", input_info("S1"), activation_func::abs));
    add_fc(g, eng, "W2", "a1", 4, impl_types::onednn);
    g.t.add(eltwise("S2", {input_info("W2"), input_info("S1")}, eltwise_mode::sum));
    g.t.add(eltwise("Zp", {input_info("O"), input_info("S2")}, eltwise_mode::prod));
    add_fc(g, eng, "Zb", "S2", 5, impl_types::ocl);   // second user of S2: Zp is not fused into W2
    g.t.add(reorder("out_Zb", input_info("Zb"), format::bfyx, data_types::f32));
    add_fc(g, eng, "Z", "Zp", 2, impl_types::ocl);
    g.t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.outs = {"out_Z", "out_Zb"};
    g.writers = {"W1", "W2"};
    return g;
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

void expect_same_frozen_lists(program& a, program& b) {
    for (auto* n : a.get_processing_order()) {
        if (!b.has_node(n->id())) {
            continue;
        }
        const auto& la = n->get_fused_primitives_onednn();
        const auto& lb = b.get_node(n->id()).get_fused_primitives_onednn();
        ASSERT_EQ(la.size(), lb.size()) << n->id();
        for (size_t i = 0; i < la.size(); i++) {
            EXPECT_EQ(la[i].op_type, lb[i].op_type) << n->id() << " #" << i;
            EXPECT_EQ(la[i].mem_offset, lb[i].mem_offset) << n->id() << " #" << i;
            EXPECT_EQ(la[i].mem_dep, lb[i].mem_dep) << n->id() << " #" << i;
            EXPECT_EQ(la[i].tag, lb[i].tag) << n->id() << " #" << i;
            EXPECT_EQ(la[i].flatten, lb[i].flatten) << n->id() << " #" << i;
            EXPECT_EQ(la[i].dims, lb[i].dims) << n->id() << " #" << i;
            EXPECT_EQ(la[i].dt, lb[i].dt) << n->id() << " #" << i;
        }
    }
}

// builds with the switch on, asserts formation (SKIP otherwise) and returns the program
program::ptr build_formed(engine& eng, const fc_graph& g, bool resolution_on, bool& skipped, std::string& why_not) {
    bool formed = true;
    auto prog = build_staged(eng, g, resolution_on, formation_hook(g, formed, why_not));
    skipped = !formed;
    return prog;
}

}  // namespace

// ---- E-late: row 11 on FC, the late reader of the view's owner forces the downgrade (E1..E4 shape with FC, bfyx) -------------------------
TEST(inplace_conflict_resolution, E_row11_fc_late_reader_downgrades_and_outputs_match) {
    P1A_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = row11_fc_graph(eng, /*late=*/true);
    bool skipped = false;
    std::string why_not;
    auto prog = build_formed(eng, g, true, skipped, why_not);
    if (skipped) {
        GTEST_SKIP() << "construction not formed: no formed sum at the freeze for" << why_not;
    }
    if (!prog->get_node("V").can_be_optimized()) {
        GTEST_SKIP() << "construction not formed: V is not a view of O";
    }
    const auto ax = exec_axis::build(*prog);
    REQUIRE_FORMED(ax.valid() && before(ax, *prog, "W", "Zp"), "Zp must run after the writer W");
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    EXPECT_EQ(rep->skipped_reason, "");
    ASSERT_EQ(rep->entries.size(), 1u) << rep->summary();
    EXPECT_EQ(rep->entries[0].writer_id, "W");
    EXPECT_EQ(rep->entries[0].why, war_downgrade_reason::unsafe);
    EXPECT_EQ(rep->downgrade_bits.count(wr_late_reader), 1u) << rep->summary();
    EXPECT_EQ(rep->downgrade_bits.at(wr_late_reader), 1u);
    expect_downgraded(*prog, "W", outer_dep(*prog, "W"));
    network net(prog, 0);
    const auto got = run_net(net, eng, g);
    EXPECT_EQ(mismatches(got, run_reference(eng, g)), 0u);
}

// E6 (distinguishing): the same graph with the switch off keeps the sum and the late reader sees the overwritten data.
TEST(inplace_conflict_resolution, E6_row11_fc_late_reader_with_switch_off_is_wrong) {
    P1A_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = row11_fc_graph(eng, /*late=*/true);
    bool skipped = false;
    std::string why_not;
    auto prog = build_formed(eng, g, false, skipped, why_not);
    if (skipped) {
        GTEST_SKIP() << "construction not formed: no formed sum at the freeze for" << why_not;
    }
    if (!prog->get_node("V").can_be_optimized()) {
        GTEST_SKIP() << "construction not formed: V is not a view of O";
    }
    const auto ax = exec_axis::build(*prog);
    REQUIRE_FORMED(ax.valid() && before(ax, *prog, "W", "Zp"), "Zp must run after the writer W");
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    EXPECT_EQ(rep->skipped_reason, "disabled");
    EXPECT_TRUE(rep->entries.empty());
    expect_kept(*prog, "W", outer_dep(*prog, "W"));
    network net(prog, 0);
    const auto got = run_net(net, eng, g);
    EXPECT_GT(mismatches(got, run_reference(eng, g)), 0u) << "with the switch off the late reader must observe the overwritten owner (F4 row 11)";
}

// ---- E-early: the other reader runs BEFORE the writer: the sum is kept (distinguishing: not "owner has another reader => reject") -------
TEST(inplace_conflict_resolution, E_row11_fc_early_reader_keeps_the_sum) {
    P1A_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = row11_fc_graph(eng, /*late=*/false);
    bool skipped = false;
    std::string why_not;
    auto prog = build_formed(eng, g, true, skipped, why_not);
    if (skipped) {
        GTEST_SKIP() << "construction not formed: no formed sum at the freeze for" << why_not;
    }
    if (!prog->get_node("V").can_be_optimized()) {
        GTEST_SKIP() << "construction not formed: V is not a view of O";
    }
    const auto ax = exec_axis::build(*prog);
    REQUIRE_FORMED(ax.valid() && before(ax, *prog, "R", "W"), "R must run before the writer W");
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    EXPECT_TRUE(rep->entries.empty()) << rep->summary();
    EXPECT_EQ(rep->kept, rep->candidates);
    expect_kept(*prog, "W", outer_dep(*prog, "W"));
    network net(prog, 0);
    EXPECT_EQ(mismatches(run_net(net, eng, g), run_reference(eng, g)), 0u);
}

// ---- E5 (F29 / row 11b) on FC ---------------------------------------------------------------------------------------------------------
TEST(inplace_conflict_resolution, E5_row11b_fc_ancestor_view_late_reader_downgrades) {
    P1A_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = row11b_fc_graph(eng);
    bool skipped = false;
    std::string why_not;
    auto prog = build_formed(eng, g, true, skipped, why_not);
    if (skipped) {
        GTEST_SKIP() << "construction not formed: the oneDNN sum into A was not created for" << why_not;
    }
    if (!prog->get_node("R").can_be_optimized()) {
        GTEST_SKIP() << "construction not formed: R is not a view of A";
    }
    const auto ax = exec_axis::build(*prog);
    REQUIRE_FORMED(ax.valid() && before(ax, *prog, "Cc", "Y"), "Y must run after the writer Cc");
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    ASSERT_EQ(rep->entries.size(), 1u) << rep->summary();
    EXPECT_EQ(rep->entries[0].writer_id, "Cc");
    EXPECT_EQ(rep->downgrade_bits.count(wr_late_reader), 1u) << rep->summary();
    expect_downgraded(*prog, "Cc", outer_dep(*prog, "Cc"));
    network net(prog, 0);
    EXPECT_EQ(mismatches(run_net(net, eng, g), run_reference(eng, g)), 0u);
}

// ---- C2b: a MIDDLE link of a chain is downgraded, the next link stays; D2 / D6 (Debug asserts) pass, the new root is not poolable ---------
TEST(inplace_conflict_resolution, C2b_middle_link_downgraded_new_root_is_not_poolable_and_outputs_match) {
    P1A_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = chain_mid_graph(eng);
    bool skipped = false;
    std::string why_not;
    std::string build_error;
    program::ptr prog;
    try {
        prog = build_formed(eng, g, true, skipped, why_not);   // Debug: D1 / I-1a-4 / D2 / D6 assertions run inside
    } catch (const std::exception& e) {
        build_error = e.what();
    }
    ASSERT_TRUE(build_error.empty()) << "D2 / D6 / I-1a-4 must pass on a downgraded middle link: " << build_error;
    if (skipped) {
        GTEST_SKIP() << "construction not formed: no formed sum at the freeze for" << why_not;
    }
    if (!prog->get_node("V").can_be_optimized()) {
        GTEST_SKIP() << "construction not formed: V is not a view of O";
    }
    const auto ax = exec_axis::build(*prog);
    REQUIRE_FORMED(ax.valid() && before(ax, *prog, "W1", "W2") && before(ax, *prog, "W2", "Zp"), "expected W1 < W2 < Zp");
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    ASSERT_EQ(rep->entries.size(), 1u) << rep->summary();
    EXPECT_EQ(rep->entries[0].writer_id, "W1") << rep->summary();
    EXPECT_EQ(rep->kept, 1u);
    expect_downgraded(*prog, "W1", outer_dep(*prog, "W1"));
    expect_kept(*prog, "W2", outer_dep(*prog, "W2"));
    // W1 is now the root of the {W1 -> W2} chain: not poolable (basic_memory_dependencies marks the root and the writer)
    EXPECT_FALSE(prog->get_node("W1").can_share_buffer()) << "the new chain root must stay off the pool";
    EXPECT_FALSE(prog->get_node("W2").can_share_buffer());
    network net(prog, 0);
    EXPECT_EQ(mismatches(run_net(net, eng, g), run_reference(eng, g)), 0u);
}

// ---- E7: Safe neutrality: switch on and off give identical frozen lists, restrictions, can_share_buffer and bit-identical outputs -------
namespace {
void run_e7(const fc_graph& g) {
    auto& eng = get_test_engine();
    bool skipped = false;
    std::string why_not;
    auto on = build_formed(eng, g, true, skipped, why_not);
    if (skipped) {
        GTEST_SKIP() << "construction not formed: no formed sum at the freeze for" << why_not;
    }
    bool skipped2 = false;
    auto off = build_formed(eng, g, false, skipped2, why_not);
    ASSERT_FALSE(skipped2);
    const auto* rep = on->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    EXPECT_TRUE(rep->entries.empty()) << rep->summary();
    EXPECT_EQ(rep->kept, rep->candidates);
    EXPECT_GT(rep->candidates, 0u);
    expect_same_frozen_lists(*on, *off);
    for (auto* n : on->get_processing_order()) {
        ASSERT_TRUE(off->has_node(n->id())) << n->id();
        auto& m = off->get_node(n->id());
        EXPECT_EQ(n->get_memory_dependencies(), m.get_memory_dependencies()) << n->id();
        EXPECT_EQ(n->can_share_buffer(), m.can_share_buffer()) << n->id();
    }
    network net_on(on, 0), net_off(off, 0);
    EXPECT_EQ(run_net(net_on, eng, g), run_net(net_off, eng, g)) << "switch on/off outputs must be bit-identical on a Safe graph";
}
}  // namespace

TEST(inplace_conflict_resolution, E7_safe_graph_is_neutral_base) {
    P1A_SKIP_IF_NO_IMMAD();
    run_e7(base_graph(get_test_engine()));
}

TEST(inplace_conflict_resolution, E7_safe_graph_is_neutral_residual_chain_depth3) {
    P1A_SKIP_IF_NO_IMMAD();
    run_e7(chain_graph(get_test_engine()));
}

// ---- E8: cache export / import after a downgrade and after a kept sum -----------------------------------------------------------------
namespace {
void run_e8(const fc_graph& g, const std::string& writer, bool expect_downgrade) {
    auto& eng = get_test_engine();
    bool skipped = false;
    std::string why_not;
    auto prog = build_formed(eng, g, true, skipped, why_not);
    if (skipped) {
        GTEST_SKIP() << "construction not formed: no formed sum at the freeze for" << why_not;
    }
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    REQUIRE_FORMED(rep->entries.empty() == !expect_downgrade, "the scenario did not produce the expected downgrade state: " + rep->summary());
    auto cfg = opt_cfg(eng, g, true);
    program::ptr imported;
    ASSERT_NO_THROW(imported = roundtrip(eng, *prog, cfg));
    ASSERT_NE(imported, nullptr);
    expect_same_frozen_lists(*prog, *imported);
    if (expect_downgrade) {
        expect_downgraded(*imported, writer, outer_dep(*imported, writer));
    } else {
        expect_kept(*imported, writer, outer_dep(*imported, writer));
    }
    network net(imported);
    EXPECT_EQ(mismatches(run_net(net, eng, g), run_reference(eng, g)), 0u);
}
}  // namespace

TEST(inplace_conflict_resolution, E8_cache_roundtrip_after_downgrade) {
    P1A_SKIP_IF_NO_IMMAD();
    run_e8(row11_fc_graph(get_test_engine(), true), "W", true);
}

TEST(inplace_conflict_resolution, E8_cache_roundtrip_kept_sum) {
    P1A_SKIP_IF_NO_IMMAD();
    run_e8(base_graph(get_test_engine()), "W", false);
}

// ---- E9: the analysis stage throws once. Release: every candidate is downgraded (analysis_exception), outputs are right. Debug: rethrow. -
TEST(inplace_conflict_resolution, E9_analysis_failure_fallback) {
    P1A_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = base_graph(eng);
    set_inplace_conflict_fault_for_test(true);
#    ifdef NDEBUG
    bool skipped = false;
    std::string why_not;
    auto prog = build_formed(eng, g, true, skipped, why_not);
    set_inplace_conflict_fault_for_test(false);
    if (skipped) {
        GTEST_SKIP() << "construction not formed: no formed sum at the freeze for" << why_not;
    }
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    ASSERT_EQ(rep->entries.size(), 1u) << rep->summary();
    EXPECT_EQ(rep->entries[0].why, war_downgrade_reason::analysis_exception);
    expect_downgraded(*prog, "W", outer_dep(*prog, "W"));
    network net(prog, 0);
    EXPECT_EQ(mismatches(run_net(net, eng, g), run_reference(eng, g)), 0u);
#    else
    EXPECT_ANY_THROW(build_staged(eng, g, true));
    EXPECT_FALSE(take_inplace_conflict_fault_for_test()) << "the one-shot hook must have been consumed by the failed build";
    set_inplace_conflict_fault_for_test(false);
#    endif
}

// The hook is one-shot: it does not leak into the next program (audit C3).
TEST(inplace_conflict_resolution, E9b_fault_hook_is_one_shot) {
    set_inplace_conflict_fault_for_test(true);
    EXPECT_TRUE(take_inplace_conflict_fault_for_test());
    EXPECT_FALSE(take_inplace_conflict_fault_for_test());
}

// ---- E10: a true OOOQ program (use_onednn off): no candidates, restrictions bit-identical with the switch on and off --------------------
TEST(inplace_conflict_resolution, E10_oooq_and_in_order_without_onednn_are_neutral) {
    auto& eng = get_test_engine();
    for (auto q : {QueueTypes::out_of_order, QueueTypes::in_order}) {
        auto make = [&](bool on) {
            ExecutionConfig config = get_test_default_config(eng);
            config.set_property(ov::intel_gpu::optimize_data(true));
            config.set_property(ov::intel_gpu::queue_type(q));
            config.set_property(ov::intel_gpu::enable_inplace_conflict_resolution(on));
            // a genuine out-of-order queue needs use_onednn(false) as a *user* property (finalize_impl forces in_order otherwise)
            config.set_user_property({ov::intel_gpu::use_onednn(false)}, ov::OptionVisibility::RELEASE_INTERNAL);
            topology t;
            t.add(input_layout("a", lay(kA)));
            t.add(input_layout("b", lay(kA)));
            t.add(activation("p", input_info("a"), activation_func::relu));
            t.add(activation("q", input_info("b"), activation_func::relu));
            t.add(eltwise("s", {input_info("p"), input_info("q")}, eltwise_mode::sum));
            t.add(activation("r", input_info("s"), activation_func::abs));
            t.add(eltwise("o", {input_info("r"), input_info("p")}, eltwise_mode::prod));
            return program::build_program(eng, t, config, false, false);
        };
        auto on = make(true);
        auto off = make(false);
        EXPECT_EQ(on->get_config().get_queue_type(), q) << "use_onednn(false) must keep the requested queue type";
        const auto* rep = on->get_inplace_conflict_report();
        ASSERT_NE(rep, nullptr);
        EXPECT_EQ(rep->skipped_reason, "no_candidates") << rep->summary();
        EXPECT_EQ(rep->candidates, 0u);
        for (auto* n : on->get_processing_order()) {
            ASSERT_TRUE(off->has_node(n->id())) << n->id();
            EXPECT_EQ(n->get_memory_dependencies(), off->get_node(n->id()).get_memory_dependencies()) << n->id();
            EXPECT_EQ(n->can_share_buffer(), off->get_node(n->id()).can_share_buffer()) << n->id();
        }
    }
}

// ---- E11: dynamic nodes never carry a frozen in-place sum, so the dynamic program has no candidates --------------------------------------
TEST(inplace_conflict_resolution, E11_dynamic_program_has_no_candidates) {
    auto& eng = get_test_engine();
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    topology t;
    t.add(input_layout("a", layout{ov::PartialShape{-1, 16}, data_types::f16, format::bfyx}));
    t.add(input_layout("b", layout{ov::PartialShape{-1, 16}, data_types::f16, format::bfyx}));
    t.add(activation("p", input_info("a"), activation_func::relu));
    t.add(eltwise("s", {input_info("p"), input_info("b")}, eltwise_mode::sum));
    t.add(activation("r", input_info("s"), activation_func::abs));
    auto prog = program::build_program(eng, t, config, false, false);
    ASSERT_TRUE(prog->get_node("s").is_dynamic());
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    EXPECT_EQ(rep->candidates, 0u) << rep->summary();
    EXPECT_TRUE(rep->entries.empty());
}

// ---- E12: frozen / live divergence after the freeze (X becomes a user-visible output): F5 downgrades first, 1a sees no candidate; the three
//      sides (frozen entry, impl slot, get_reused) agree and the output is right ------------------------------------------------------------
TEST(inplace_conflict_resolution, E12_divergence_after_freeze_three_sides_agree) {
    P1A_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = base_graph(eng);
    bool formed = true;
    std::string why_not;
    int32_t dep = -1;
    auto hook = [&](program& p) {
        formation_hook(g, formed, why_not)(p);
        if (formed) {
            dep = sum_dep_if_formed(p.get_node("W"));
            p.get_node("X").set_output(true);   // post-freeze divergence: the sum would overwrite a user-visible buffer
        }
    };
    auto prog = build_staged(eng, g, true, hook);
    if (!formed) {
        GTEST_SKIP() << "construction not formed: no formed sum at the freeze for" << why_not;
    }
    ASSERT_GE(dep, 0);
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    EXPECT_EQ(rep->candidates, 0u) << "F5 must have downgraded the sum before the 1a planner: " << rep->summary();
    expect_downgraded(*prog, "W", dep);
    network net(prog, 0);
    EXPECT_EQ(mismatches(run_net(net, eng, g), run_reference(eng, g)), 0u);
}

// =====================================================================================================================================
// Slice 1a-2: the binary_mul_inplace veto (frozen entry relabelled binary_mul + attrs slot algorithm swapped, no new field).
// =====================================================================================================================================
namespace {

// A = abs(ia); W = FC_onednn(in1); M = prod(W, A) [fused into W as binary_mul_inplace]; Z = FC_ocl(M); out_Z.
fc_graph mul_graph(engine& eng) {
    fc_graph g;
    g.t.add(input_layout("in1", f16_lay()));
    g.t.add(input_layout("ia", f16_lay()));
    g.inputs = {{"in1", f16_lay()}, {"ia", f16_lay()}};
    g.t.add(activation("A", input_info("ia"), activation_func::abs));
    add_fc(g, eng, "W", "in1", 0, impl_types::onednn);
    g.t.add(eltwise("M", {input_info("W"), input_info("A")}, eltwise_mode::prod));
    add_fc(g, eng, "Z", "M", 2, impl_types::ocl);
    g.t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.outs = {"out_Z"};
    return g;
}

// outer dependency of the fused prod of `w` if the mul_inplace is formed (frozen entry AND live predicate), else -1
int32_t mul_dep_if_formed(const program_node& w) {
    for (const auto& fd : w.get_fused_primitives()) {
        if (fd.is_type<eltwise>() && fd.typed_desc<eltwise>()->mode == eltwise_mode::prod && fd.deps.size() == 1 && fd.has_outer_dep()) {
            const int32_t dep = fd.outer_dep_start_idx;
            if (fusing::is_frozen_inplace_post_op(w, dep, eltwise_mode::prod) && fusing::get_reused_eltwmem_idx(w) == dep) {
                return dep;
            }
            return -1;
        }
    }
    return -1;
}

std::vector<dnnl::algorithm> binary_algs(program_node& n) {
    std::vector<dnnl::algorithm> r;
    const auto ops = n.get_onednn_primitive_attributes()->get_post_ops();
    for (int i = 0; i < ops.len(); i++) {
        if (ops.kind(i) == dnnl::primitive::kind::binary) {
            dnnl::algorithm alg;
            dnnl::memory::desc md;
            ops.get_params_binary(i, alg, md);
            r.push_back(alg);
        }
    }
    return r;
}

bool has_alg(const std::vector<dnnl::algorithm>& v, dnnl::algorithm a) {
    return std::find(v.begin(), v.end(), a) != v.end();
}

bool frozen_has(const program_node& n, onednn_post_op_type t) {
    for (const auto& e : n.get_fused_primitives_onednn()) {
        if (e.op_type == t) {
            return true;
        }
    }
    return false;
}

// a hand-written plan that vetoes the mul_inplace of W (the real planner never does on a real graph: M0/M1 make the version set {P_m})
inplace_conflict_plan mul_veto_plan(program& p, int32_t dep) {
    inplace_conflict_plan plan;
    inplace_conflict_action a;
    a.writer = &p.get_node("W");
    a.mem_dep = dep;
    a.origin = edge_origin::onednn_mul;
    a.why = war_downgrade_reason::unsafe;
    a.r.reasons = wr_late_reader;
    plan.actions.push_back(std::move(a));
    return plan;
}

}  // namespace

// M1: a real FC mul_inplace is planned and judged Safe: kept, frozen entry and impl untouched (zero behaviour change on real graphs).
TEST(inplace_conflict_resolution, M1_real_mul_inplace_is_planned_safe_and_kept) {
    P1A_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = mul_graph(eng);
    int32_t dep = -1;
    auto hook = [&](program& p) {
        dep = mul_dep_if_formed(p.get_node("W"));
    };
    auto prog = build_staged(eng, g, true, hook);
    if (dep < 0) {
        GTEST_SKIP() << "construction not formed: no frozen+live binary_mul_inplace on W at the freeze";
    }
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    EXPECT_EQ(rep->candidates, 1u) << rep->summary();
    EXPECT_EQ(rep->kept, 1u);
    EXPECT_TRUE(rep->entries.empty()) << rep->summary();
    auto& W = prog->get_node("W");
    EXPECT_TRUE(frozen_has(W, onednn_post_op_type::binary_mul_inplace));
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W), dep);
    EXPECT_TRUE(has_alg(binary_algs(W), dnnl::algorithm::binary_mul_inplace));
    // the planner itself: one onednn_mul action, Safe
    const auto aa = alias_analysis::build(*prog);
    const auto axis = exec_axis::build(*prog);
    ASSERT_TRUE(axis.valid());
    const auto plan = plan_inplace_conflict_resolution(aa, axis);
    ASSERT_EQ(plan.actions.size(), 1u);
    EXPECT_EQ(plan.actions[0].origin, edge_origin::onednn_mul);
    EXPECT_EQ(plan.actions[0].why, war_downgrade_reason::none);
    network net(prog, 0);
    EXPECT_EQ(mismatches(run_net(net, eng, g), run_reference(eng, g)), 0u);
}

// M2: the executor vetoes a mul_inplace from a hand-written plan: frozen entry binary_mul, attrs slot algorithm binary_mul, impl rebuilt, no alias
// at runtime (G-impl), output equal to the reference, and the state survives an export / import round trip.
TEST(inplace_conflict_resolution, M2_mul_veto_relabels_entry_swaps_alg_and_roundtrips) {
    P1A_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = mul_graph(eng);
    int32_t dep = -1;
    size_t algs_before_inplace = 0;
    const primitive_impl* impl_before = nullptr;
    inplace_conflict_report rep;
    auto hook = [&](program& p) {
        dep = mul_dep_if_formed(p.get_node("W"));
        if (dep < 0) {
            return;
        }
        auto& W = p.get_node("W");
        algs_before_inplace = has_alg(binary_algs(W), dnnl::algorithm::binary_mul_inplace) ? 1 : 0;
        impl_before = W.get_selected_impl();
        refreeze_onednn_inplace_post_ops::apply_inplace_conflict_plan(p, mul_veto_plan(p, dep), rep);
    };
    auto prog = build_staged(eng, g, /*resolution_on=*/false, hook);   // switch off: only the hand-written plan acts
    if (dep < 0) {
        GTEST_SKIP() << "construction not formed: no frozen+live binary_mul_inplace on W at the freeze";
    }
    ASSERT_EQ(algs_before_inplace, 1u) << "the compiled chain must hold a binary_mul_inplace before the veto";
    ASSERT_EQ(rep.entries.size(), 1u);
    EXPECT_EQ(rep.entries[0].origin, edge_origin::onednn_mul);
    auto& W = prog->get_node("W");
    EXPECT_FALSE(frozen_has(W, onednn_post_op_type::binary_mul_inplace));
    EXPECT_TRUE(frozen_has(W, onednn_post_op_type::binary_mul));
    const auto algs = binary_algs(W);
    EXPECT_TRUE(has_alg(algs, dnnl::algorithm::binary_mul)) << "attrs slot algorithm must be binary_mul";
    EXPECT_FALSE(has_alg(algs, dnnl::algorithm::binary_mul_inplace));
    EXPECT_NE(W.get_selected_impl(), impl_before) << "the oneDNN impl must have been rebuilt";
    ASSERT_TRUE(W.get_selected_impl()->is_onednn());
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W), -1);
    {
        const auto slots = get_compiled_post_op_slots(W.get_fused_primitives_onednn(), static_cast<size_t>(dep), {onednn_post_op_type::binary_mul});
        ASSERT_TRUE(slots.slot.has_value());
        EXPECT_FALSE(W.get_selected_impl()->is_binary_mul_inplace_post_op(slots.kinds, *slots.slot));
    }
    network net(prog, 0);
    EXPECT_FALSE(net.get_primitive("W")->reused_eltw_alias_allowed(dep)) << "G-impl: no alias after the veto";
    EXPECT_EQ(net.get_primitive("W")->get_reused_eltw_dep_idx(), -1);
    EXPECT_FALSE(eng.is_the_same_buffer(*net.get_primitive("W")->output_memory_ptr(), *net.get_primitive("A")->output_memory_ptr()));
    EXPECT_EQ(mismatches(run_net(net, eng, g), run_reference(eng, g)), 0u);

    // export / import: the vetoed state is consistent after the round trip (the import check does not throw, the compiled alg is binary_mul)
    auto cfg = opt_cfg(eng, g, false);
    program::ptr imported;
    ASSERT_NO_THROW(imported = roundtrip(eng, *prog, cfg));
    ASSERT_NE(imported, nullptr);
    expect_same_frozen_lists(*prog, *imported);
    auto& Wi = imported->get_node("W");
    EXPECT_TRUE(frozen_has(Wi, onednn_post_op_type::binary_mul));
    EXPECT_FALSE(frozen_has(Wi, onednn_post_op_type::binary_mul_inplace));
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(Wi), -1);
    network net_i(imported);
    EXPECT_EQ(mismatches(run_net(net_i, eng, g), run_reference(eng, g)), 0u);
}

// M3: a non-oneDNN impl (OCL under a oneDNN-preferred node, F17 G-impl) only gets its frozen label changed: attrs and impl are untouched.
TEST(inplace_conflict_resolution, M3_mul_veto_on_non_onednn_impl_only_relabels) {
    P1A_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = mul_graph(eng);
    int32_t dep = -1;
    bool ocl_swapped = false;
    const primitive_impl* impl_before = nullptr;
    const void* attrs_before = nullptr;
    inplace_conflict_report rep;
    auto hook = [&](program& p) {
        dep = mul_dep_if_formed(p.get_node("W"));
        if (dep < 0) {
            return;
        }
        auto& W = p.get_node("W");
        auto params = W.get_kernel_impl_params();
        std::unique_ptr<primitive_impl> ocl_impl;
        for (const auto& m : W.type()->get_all_implementations()) {
            if (m->get_impl_type() == impl_types::ocl) {
                ocl_impl = m->create(W, *params);
                break;
            }
        }
        if (!ocl_impl) {
            return;
        }
        W.set_selected_impl(std::move(ocl_impl));
        auto new_params = W.get_kernel_impl_params();
        p.get_kernels_cache().add_kernels_source(*new_params, W.get_selected_impl()->get_kernels_source());
        ocl_swapped = !W.get_selected_impl()->is_onednn();
        impl_before = W.get_selected_impl();
        attrs_before = W.get_onednn_primitive_attributes().get();
        refreeze_onednn_inplace_post_ops::apply_inplace_conflict_plan(p, mul_veto_plan(p, dep), rep);
    };
    auto prog = build_staged(eng, g, /*resolution_on=*/false, hook);
    if (dep < 0) {
        GTEST_SKIP() << "construction not formed: no frozen+live binary_mul_inplace on W at the freeze";
    }
    if (!ocl_swapped) {
        GTEST_SKIP() << "construction not formed: no OCL fully_connected implementation to put under the oneDNN-preferred node";
    }
    auto& W = prog->get_node("W");
    EXPECT_EQ(W.get_selected_impl(), impl_before) << "a non-oneDNN impl must not be rebuilt";
    EXPECT_FALSE(W.get_selected_impl()->is_onednn());
    EXPECT_EQ(W.get_onednn_primitive_attributes().get(), attrs_before) << "attrs of a non-oneDNN node must not be touched";
    EXPECT_FALSE(frozen_has(W, onednn_post_op_type::binary_mul_inplace));
    EXPECT_TRUE(frozen_has(W, onednn_post_op_type::binary_mul));
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W), -1);
    network net(prog, 0);
    EXPECT_FALSE(net.get_primitive("W")->reused_eltw_alias_allowed(dep));
    EXPECT_EQ(mismatches(run_net(net, eng, g), run_reference(eng, g)), 0u);
}

#endif  // ENABLE_ONEDNN_FOR_GPU
