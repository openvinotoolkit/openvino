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

#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "alias_analysis.h"
#include "alias_validation.h"
#include "exec_axis.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/kv_cache.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/op/interpolate.hpp"
#include "program_node.h"
#include "program_wrapper.h"

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
