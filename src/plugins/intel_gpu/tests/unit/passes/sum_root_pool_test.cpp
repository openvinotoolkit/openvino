// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// P1-C-0 (behaviour neutral): the census of oneDNN in-place sum classes and its white-box mask (L1S rules).
// Design: /home/ceciliapeng/repos/reports/memman/layer1/18_p1c_design.zh-CN.md (r2) sections 3.2, 3.5, 3.8 and 7 (C0-1 ... C0-9).
//
// "Hand" graphs (no optimization passes, hypothetical in-place edges, injected marks) cover the pure mask function without a oneDNN device.
// "Real" graphs (FC onednn + fused sum) are GTEST_SKIPped when the device has no oneDNN support or when the sum did not form (a SKIP counts as
// "not covered"). With enable_sum_root_pool the pass only reports: flags, restrictions and pool fields must be bit-identical to the switch off.
//
// Not in this slice (belong to P1-C-1 and need the deferred-mark members of program.hpp / L3-V): C0-6 (deferred marks guard on every exit)
// and C0-8 (L3-V white box).

#include "test_utils.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "alias_analysis.h"
#include "alias_validation.h"
#include "class_level_memory_dependencies.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "primitive_inst.h"
#include "program_helpers.h"
#include "program_node.h"

using namespace cldnn;
using namespace ::tests;

namespace {

using fusing = onednn_eltwise_fusing_helpers;

layout lay(const ov::PartialShape& s) {
    return layout{s, data_types::f32, format::bfyx};
}

const ov::PartialShape kA{1, 4, 8, 8};

ExecutionConfig hand_cfg(engine& eng, const std::vector<std::string>& outputs) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    config.set_property(ov::intel_gpu::enable_sum_root_pool(true));
    if (!outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(outputs));
    }
    return config;
}

program::ptr make_hand(engine& eng, const topology& t, const std::vector<std::string>& outs, bool internal = false, bool body = false,
                       const std::function<void(ExecutionConfig&)>& tweak = {}) {
    auto cfg = hand_cfg(eng, outs);
    if (tweak) {
        tweak(cfg);
    }
    auto prog = program::build_program(eng, t, cfg, internal, /*no_optimizations=*/true, body);
    for (auto* n : prog->get_processing_order()) {
        n->get_output_layouts(false);
        n->set_unique_id();
    }
    return prog;
}

// in -> R -> W1(sum, f1) -> W2(sum, f2) -> Z (output); every node f32 {1,4,8,8}.
topology chain_topology() {
    topology t;
    t.add(input_layout("in", lay(kA)));
    t.add(input_layout("f1", lay(kA)));
    t.add(input_layout("f2", lay(kA)));
    t.add(activation("R", input_info("in"), activation_func::relu));
    t.add(eltwise("W1", {input_info("R"), input_info("f1")}, eltwise_mode::sum));
    t.add(eltwise("W2", {input_info("W1"), input_info("f2")}, eltwise_mode::sum));
    t.add(activation("Z", input_info("W2"), activation_func::abs));
    return t;
}

struct hand_chain {
    program::ptr prog;
    std::vector<hypothetical_inplace> extra;
    std::unique_ptr<alias_analysis> aa;
    class_level::sum_root_marks_index marks;
    const alias_class* k = nullptr;
    bool ok = false;
    std::string why_not;
};

// Hypothetical in-place edges W1 dep 0 and W2 dep 0; the "natural" marks of the chain R -> W1 -> W2 (design 18 section 3.1).
std::unique_ptr<hand_chain> make_hand_chain(engine& eng, const std::function<void(program&)>& pre_aa = {}) {
    auto h = std::make_unique<hand_chain>();
    auto t = chain_topology();
    h->prog = make_hand(eng, t, {"Z"});
    auto& p = *h->prog;
    if (pre_aa) {
        pre_aa(p);
    }
    h->extra = {{&p.get_node("W1"), 0}, {&p.get_node("W2"), 0}};
    h->aa = std::make_unique<alias_analysis>(alias_analysis::build(p, compile_time_envelope{}, &h->extra));
    if (!h->aa->rejected_hypotheticals().empty()) {
        h->why_not = "a hypothetical edge was rejected";
        return h;
    }
    h->k = h->aa->class_of(p.get_node("R"));
    if (h->k == nullptr || h->k->members.size() != 3 || h->k->root_node != &p.get_node("R")) {
        h->why_not = "class of R is not {R, W1, W2}";
        return h;
    }
    p.get_node("W1").can_share_buffer(false);
    p.get_node("W2").can_share_buffer(false);
    auto* R = &p.get_node("R");
    auto* W1 = &p.get_node("W1");
    auto* W2 = &p.get_node("W2");
    h->marks.of_writer[W1] = {R, R};
    h->marks.of_writer[W2] = {W1, W1};
    h->marks.writers_marking[R] = {W1};
    h->marks.writers_marking[W1] = {W2};
    h->ok = true;
    return h;
}

#define REQUIRE_FORMED(cond, msg)                                       \
    do {                                                                \
        if (!(cond)) {                                                  \
            GTEST_SKIP() << "construction not formed: " << (msg);       \
        }                                                               \
    } while (0)

const auto kNeverAmbiguous = [](uint32_t) { return false; };

// the mask without the program level bit (hand programs have no 1a report, so L1S-0 always fires there)
uint32_t mask_of(const hand_chain& h, const alias_class& k, const class_level::sum_root_marks_index& marks,
                 const std::function<bool(uint32_t)>& amb = kNeverAmbiguous, bool admit_padded = true) {
    const uint32_t m = class_level::sum_class_reject_mask(k, *h.aa, *h.prog, marks, amb, /*include_hypothetical=*/true, admit_padded);
    EXPECT_NE(m & class_level::rj_program_level, 0u) << "a hand program has no 1a report: L1S-0 must fire";
    return m & ~static_cast<uint32_t>(class_level::rj_program_level);
}

}  // namespace

// ===============================================================================================================================
// C0-1: every L1S rule once, hypothetical edges and injected marks. Includes the 2-writer chain (BL-1) as the positive case.
// ===============================================================================================================================
TEST(sum_root_pool_census, C0_1_mask_white_box_hypothetical_positive_two_writer_chain) {
    auto& eng = get_test_engine();
    auto h = make_hand_chain(eng);
    REQUIRE_FORMED(h->ok, h->why_not);
    EXPECT_TRUE(class_level::is_sum_candidate_class(*h->k, *h->aa, /*include_hypothetical=*/true));
    EXPECT_FALSE(class_level::is_sum_candidate_class(*h->k, *h->aa, /*include_hypothetical=*/false)) << "hypothetical edges are not onednn_sum edges";
    EXPECT_EQ(mask_of(*h, *h->k, h->marks), 0u) << class_level::to_string(mask_of(*h, *h->k, h->marks));
}

TEST(sum_root_pool_census, C0_1_mask_white_box_hypothetical_provenance_negatives) {
    auto& eng = get_test_engine();
    auto h = make_hand_chain(eng);
    REQUIRE_FORMED(h->ok, h->why_not);
    auto& p = *h->prog;
    auto* R = &p.get_node("R");
    auto* W1 = &p.get_node("W1");
    auto* W2 = &p.get_node("W2");
    auto* Z = &p.get_node("Z");
    {  // (a) legacy root and owner differ (defect B / port difference)
        auto m = h->marks;
        m.of_writer[W2] = {R, W1};
        EXPECT_EQ(mask_of(*h, *h->k, m), class_level::rj_sum_provenance) << class_level::to_string(mask_of(*h, *h->k, m));
    }
    {  // (b) the writer marks a node outside the class
        auto m = h->marks;
        m.of_writer[W2] = {Z, Z};
        EXPECT_EQ(mask_of(*h, *h->k, m), class_level::rj_sum_provenance);
    }
    {  // (c) another writer (outside the class) also marks the root
        auto m = h->marks;
        m.writers_marking[R].push_back(Z);
        EXPECT_EQ(mask_of(*h, *h->k, m), class_level::rj_sum_provenance);
    }
    {  // (0) basic_memory_dependencies never marked W2
        auto m = h->marks;
        m.of_writer.erase(W2);
        EXPECT_EQ(mask_of(*h, *h->k, m), class_level::rj_sum_provenance);
    }
}

TEST(sum_root_pool_census, C0_1_mask_white_box_hypothetical_one_rule_per_bit) {
    auto& eng = get_test_engine();
    auto h = make_hand_chain(eng);
    REQUIRE_FORMED(h->ok, h->why_not);
    auto& p = *h->prog;
    auto* R = &p.get_node("R");
    auto* W1 = &p.get_node("W1");
    auto* W2 = &p.get_node("W2");
    auto* Z = &p.get_node("Z");
    using namespace class_level;
    {  // L1S-1: domain
        alias_class k = *h->k;
        k.domain |= dm_user_output;
        EXPECT_EQ(mask_of(*h, k, h->marks), rj_placement);
    }
    {  // L1S-2: root on a port != 0
        alias_class k = *h->k;
        k.root.index = 1;
        EXPECT_EQ(mask_of(*h, k, h->marks), rj_root_kind);
    }
    {  // L1S-3: a root that is non-shareable and not marked by any sum writer
        auto m = h->marks;
        m.writers_marking.erase(R);
        R->can_share_buffer(false);
        EXPECT_EQ(mask_of(*h, *h->k, m), rj_root_not_shareable);
        R->can_share_buffer(true);
        // ... but a root non-shareable only because a sum writer marked it is not reported (no deferral before P1-C-1)
        R->can_share_buffer(false);
        EXPECT_EQ(mask_of(*h, *h->k, h->marks), 0u);
        R->can_share_buffer(true);
        // an output root
        R->set_output(true);
        EXPECT_EQ(mask_of(*h, *h->k, h->marks), rj_root_not_shareable);
        R->set_output(false);
    }
    {  // L1S-4: members must be sum writers or views
        alias_class k = *h->k;
        for (auto& m : k.members) {
            if (m.node == W1) {
                m.kind = alias_kind::MAY_INPLACE;
            }
        }
        EXPECT_NE(mask_of(*h, k, h->marks) & rj_member_kind, 0u);
        alias_class k2 = *h->k;
        for (auto& m : k2.members) {
            if (m.node == W1) {
                m.kind = alias_kind::MUST_SUBREGION;   // mixed concat class
            }
        }
        EXPECT_NE(mask_of(*h, k2, h->marks) & rj_member_kind, 0u);
    }
    {  // L1S-5: a runtime skippable reader / member
        Z->set_runtime_skippable(true);
        EXPECT_EQ(mask_of(*h, *h->k, h->marks), rj_dynamic);
        Z->set_runtime_skippable(false);
        W1->set_runtime_skippable(true);
        EXPECT_NE(mask_of(*h, *h->k, h->marks) & rj_dynamic, 0u);
        W1->set_runtime_skippable(false);
    }
    {  // L1S-7: a member on a port != 0
        alias_class k = *h->k;
        k.members.back().id.index = 1;
        EXPECT_NE(mask_of(*h, k, h->marks) & rj_multi_output, 0u);
    }
    {  // L1S-8: a writer that is still shareable
        W2->can_share_buffer(true);
        EXPECT_EQ(mask_of(*h, *h->k, h->marks), rj_sum_writer);
        W2->can_share_buffer(false);
        // a writer that already carries a class flag
        W2->set_pool_class_root(true);
        EXPECT_EQ(mask_of(*h, *h->k, h->marks), rj_sum_writer);
        W2->set_pool_class_root(false);
    }
    {  // L1S-9: ambiguous uid
        const uint32_t u = static_cast<uint32_t>(W1->get_unique_id());
        EXPECT_EQ(mask_of(*h, *h->k, h->marks, [&](uint32_t x) { return x == u; }), rj_ambiguous_uid);
    }
    {  // L1S-11: a reader inside a shape_of subgraph
        Z->set_in_shape_of_subgraph(true);
        EXPECT_NE(mask_of(*h, *h->k, h->marks) & rj_shape_of_reader, 0u);
        Z->set_in_shape_of_subgraph(false);
    }
    EXPECT_EQ(mask_of(*h, *h->k, h->marks), 0u) << "all temporary mutations were undone";
    // L1S-6 (a member that does not use the pool) and L1S-10 (external root member) need node types that cannot be a hypothetical writer
    // here; they are covered by the concat counterparts (same predicates). L1S-12 has its own test below.
}

TEST(sum_root_pool_census, C0_1_mask_white_box_hypothetical_padded_root_rule_l1s_12) {
    auto& eng = get_test_engine();
    layout padded{kA, data_types::f32, format::bfyx, padding({0, 0, 1, 1}, {0, 0, 1, 1})};
    auto h = make_hand_chain(eng, [&](program& p) {
        for (const char* id : {"R", "W1", "W2"}) {
            p.get_node(id).set_output_layout(padded, false);
        }
    });
    REQUIRE_FORMED(h->ok, h->why_not);
    REQUIRE_FORMED(static_cast<bool>(h->prog->get_node("R").get_output_layout(0).data_padding), "R carries no padding");
    EXPECT_EQ(mask_of(*h, *h->k, h->marks, kNeverAmbiguous, /*admit_padded=*/true), 0u) << "kAdmitPaddedSumRoot=true admits a padded root";
    EXPECT_EQ(mask_of(*h, *h->k, h->marks, kNeverAmbiguous, /*admit_padded=*/false), class_level::rj_sum_root_padded);
    EXPECT_TRUE(class_level::kAdmitPaddedSumRoot) << "the preset ruling of NB-4 (C-16b may flip it in P1-C-1)";
}

TEST(sum_root_pool_census, C0_1_to_string_names_the_new_bits) {
    EXPECT_EQ(class_level::to_string(class_level::rj_sum_provenance), "sum_provenance");
    EXPECT_EQ(class_level::to_string(class_level::rj_sum_writer), "sum_writer");
    EXPECT_EQ(class_level::to_string(class_level::rj_sum_root_padded), "sum_root_padded");
    EXPECT_EQ(class_level::to_string(class_level::rj_sum_provenance | class_level::rj_placement), "placement|sum_provenance");
}

// ===============================================================================================================================
// C0-2: the shared helper equals the former inline logic of basic_memory_dependencies (checked on real oneDNN graphs in the tests below via
// inline_walk_equals_helper) and is a no-op without a oneDNN writer.
// ===============================================================================================================================
TEST(sum_root_pool_census, C0_2_helper_is_nullopt_without_a_onednn_writer) {
    auto& eng = get_test_engine();
    auto t = chain_topology();
    auto prog = make_hand(eng, t, {"Z"});
    for (auto* n : prog->get_processing_order()) {
        EXPECT_FALSE(fusing::sum_writer_root_marks(*n).has_value()) << n->id();
    }
    const auto marks = class_level::collect_sum_root_marks(*prog);
    EXPECT_TRUE(marks.of_writer.empty());
    EXPECT_TRUE(marks.writers_marking.empty());
}

// ===============================================================================================================================
// C0-3: a dynamic FC+add never forms a sum; the census sees no candidate (dynamic sum is always not_supported).
// ===============================================================================================================================
TEST(sum_root_pool_census, C0_3_dynamic_sum_is_not_supported_guard) {
    auto& eng = get_test_engine();
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    config.set_property(ov::intel_gpu::enable_sum_root_pool(true));
    topology t;
    t.add(input_layout("a", layout{ov::PartialShape{-1, 16}, data_types::f16, format::bfyx}));
    t.add(input_layout("b", layout{ov::PartialShape{-1, 16}, data_types::f16, format::bfyx}));
    t.add(activation("p", input_info("a"), activation_func::relu));
    t.add(eltwise("s", {input_info("p"), input_info("b")}, eltwise_mode::sum));
    t.add(activation("r", input_info("s"), activation_func::abs));
    auto prog = program::build_program(eng, t, config, false, false);
    ASSERT_TRUE(prog->get_node("s").is_dynamic());
    for (auto* n : prog->get_processing_order()) {
        for (const auto& fd : n->get_fused_primitives()) {
            if (fd.is_type<eltwise>() && fd.has_outer_dep()) {
                EXPECT_EQ(fusing::get_add_fusing_type(*n, fd), fusing::add_fusing_type::not_supported) << n->id();
            }
        }
    }
    const auto* aa = prog->get_alias_analysis();
    ASSERT_NE(aa, nullptr);
    for (const auto& e : aa->edges()) {
        EXPECT_NE(e.origin, edge_origin::onednn_sum);
    }
    const auto census = class_level::run_sum_census(*prog);
    EXPECT_EQ(census.candidates, 0u) << census.summary();
    EXPECT_EQ(census.would_promote, 0u);
    EXPECT_EQ(census.no_edge_sum, 0u);
}

// ===============================================================================================================================
// C0-4 / C0-5: the program level predicate (single implementation) and the OOOQ keep-LEGACY behaviour with the switch on.
// ===============================================================================================================================
namespace {

program::ptr build_pipeline(engine& eng, bool sum_switch, QueueTypes q, const std::function<void(ExecutionConfig&)>& tweak = {}) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::queue_type(q));
    config.set_property(ov::intel_gpu::enable_sum_root_pool(sum_switch));
    // a genuine out-of-order queue needs use_onednn(false) as a user property (finalize_impl forces in_order otherwise, T-OOOQ)
    if (q == QueueTypes::out_of_order) {
        config.set_user_property({ov::intel_gpu::use_onednn(false)}, ov::OptionVisibility::RELEASE_INTERNAL);
    }
    if (tweak) {
        tweak(config);
    }
    topology t;
    t.add(input_layout("a", lay(kA)));
    t.add(input_layout("b", lay(kA)));
    t.add(activation("p", input_info("a"), activation_func::relu));
    t.add(activation("q", input_info("b"), activation_func::relu));
    t.add(eltwise("s", {input_info("p"), input_info("q")}, eltwise_mode::sum));
    t.add(activation("r", input_info("s"), activation_func::abs));
    t.add(eltwise("o", {input_info("r"), input_info("p")}, eltwise_mode::prod));
    return program::build_program(eng, t, config, false, false);
}

void expect_same_state(program& a, program& b, const std::string& what) {
    ASSERT_EQ(a.get_processing_order().size(), b.get_processing_order().size()) << what;
    for (auto* n : a.get_processing_order()) {
        ASSERT_TRUE(b.has_node(n->id())) << what << ": " << n->id();
        auto& m = b.get_node(n->id());
        EXPECT_EQ(n->get_memory_dependencies(), m.get_memory_dependencies()) << what << ": restriction of " << n->id();
        EXPECT_EQ(n->can_share_buffer(), m.can_share_buffer()) << what << ": can_share_buffer of " << n->id();
        EXPECT_EQ(n->is_pool_class_root(), m.is_pool_class_root()) << what << ": pool_class_root of " << n->id();
        EXPECT_EQ(n->has_alias_pool_root(), m.has_alias_pool_root()) << what << ": alias_pool_root of " << n->id();
        if (n->has_alias_pool_root() && m.has_alias_pool_root()) {
            EXPECT_EQ(n->get_alias_pool_root_uid(), m.get_alias_pool_root_uid()) << what << ": alias_pool_root_uid of " << n->id();
        }
        EXPECT_EQ(n->can_be_optimized(), m.can_be_optimized()) << what << ": can_be_optimized of " << n->id();
    }
}

}  // namespace

TEST(sum_root_pool_census, C0_4_oooq_bit_identical_with_the_switch_on) {
    auto& eng = get_test_engine();
    auto on = build_pipeline(eng, true, QueueTypes::out_of_order);
    auto off = build_pipeline(eng, false, QueueTypes::out_of_order);
    ASSERT_EQ(on->get_config().get_queue_type(), QueueTypes::out_of_order) << "use_onednn(false) must keep the requested queue type";
    std::string why;
    EXPECT_FALSE(class_level::sum_root_pool_active(*on, &why));
    EXPECT_EQ(why, "oooq");
    expect_same_state(*on, *off, "oooq switch on vs off");
    // the NB-9 log line ("enable_sum_root_pool ignored: oooq") is emitted by the pass under GPU_DEBUG; its text is not captured here.
}

TEST(sum_root_pool_census, C0_5_program_level_predicate) {
    auto& eng = get_test_engine();
    std::string why;
    {  // active: in_order, switch on, 1a on, report present
        auto p = build_pipeline(eng, true, QueueTypes::in_order);
        ASSERT_NE(p->get_inplace_conflict_report(), nullptr);
        EXPECT_TRUE(class_level::sum_root_pool_active(*p, &why)) << why;
    }
    {  // switch off
        auto p = build_pipeline(eng, false, QueueTypes::in_order);
        EXPECT_FALSE(class_level::sum_root_pool_active(*p, &why));
        EXPECT_EQ(why, "switch_off");
    }
    {  // 1a off
        auto p = build_pipeline(eng, true, QueueTypes::in_order, [](ExecutionConfig& c) { c.set_property(ov::intel_gpu::enable_inplace_conflict_resolution(false)); });
        EXPECT_FALSE(class_level::sum_root_pool_active(*p, &why));
        EXPECT_EQ(why, "1a_off");
    }
    {  // memory pool off
        auto p = build_pipeline(eng, true, QueueTypes::in_order, [](ExecutionConfig& c) { c.set_property(ov::intel_gpu::enable_memory_pool(false)); });
        EXPECT_FALSE(class_level::sum_root_pool_active(*p, &why));
        EXPECT_EQ(why, "memory_pool_off");
    }
    auto t = chain_topology();
    {  // internal / body programs
        auto p = make_hand(eng, t, {"Z"}, /*internal=*/true);
        EXPECT_FALSE(class_level::sum_root_pool_active(*p, &why));
        EXPECT_EQ(why, "internal");
        auto t2 = chain_topology();
        auto b = make_hand(eng, t2, {"Z"}, /*internal=*/false, /*body=*/true);
        EXPECT_FALSE(class_level::sum_root_pool_active(*b, &why));
        EXPECT_EQ(why, "body");
    }
    {  // no 1a report (a hand program skips the optimization passes)
        auto t3 = chain_topology();
        auto p = make_hand(eng, t3, {"Z"});
        EXPECT_FALSE(class_level::sum_root_pool_active(*p, &why));
        EXPECT_EQ(why, "no_report");
    }
    // 1a_skipped:<reason> is not reachable once the internal/body/oooq/switch checks above ran first (the report only skips for those or "disabled")
}

TEST(sum_root_pool_census, C0_7a_census_only_switch_on_is_bit_identical_without_onednn_sums) {
    auto& eng = get_test_engine();
    auto on = build_pipeline(eng, true, QueueTypes::in_order);
    auto off = build_pipeline(eng, false, QueueTypes::in_order);
    expect_same_state(*on, *off, "in_order switch on vs off");
    const auto census = class_level::run_sum_census(*on);
    EXPECT_EQ(census.candidates, 0u) << census.summary();
    EXPECT_EQ(census.would_promote, 0u);
}

// ===============================================================================================================================
// C0-9: the generator (L3) closes two intersecting classes in both directions (I-B1 across classes); disjoint classes stay unrelated.
// ===============================================================================================================================
TEST(sum_root_pool_census, C0_9_ib1_cross_class_white_box) {
    using namespace class_level;
    auto make = [](bool related) {
        simple_restriction_store s;
        s.uids = {10, 11, 12, 13};   // class A: root 0, member 1; class B: root 2, member 3
        s.sets = {{}, {}, {}, {}};
        if (related) {
            s.insert(1, 13);   // member of A and member of B conflict (one direction only, as the pool sees it)
            s.insert(3, 11);
        }
        return s;
    };
    const std::vector<promoted_class> classes = {{0, {0, 1}}, {2, {2, 3}}};
    {
        auto s = make(true);
        const auto r0 = snapshot_restrictions(s);
        materialize_class_restrictions(s, classes);
        EXPECT_TRUE(s.contains(0, 12) && s.contains(2, 10)) << "the roots of two intersecting classes must restrict each other";
        EXPECT_EQ(check_class_invariants(s, r0, classes), "");
    }
    {
        auto s = make(false);
        const auto r0 = snapshot_restrictions(s);
        materialize_class_restrictions(s, classes);
        EXPECT_FALSE(s.contains(0, 12) || s.contains(2, 10)) << "disjoint classes must not become related";
        EXPECT_EQ(check_class_invariants(s, r0, classes), "");
    }
}

// ===============================================================================================================================
// Real oneDNN graphs: census, helper equivalence, and the bit-identical neutrality of the census-only switch.
// ===============================================================================================================================
#ifdef ENABLE_ONEDNN_FOR_GPU

namespace {

#define P1C0_SKIP_IF_NO_IMMAD()                                                                \
    do {                                                                                       \
        if (!get_test_engine().get_device_info().supports_immad) {                             \
            GTEST_SKIP() << "the oneDNN in-place sum needs a oneDNN device (supports_immad)";  \
        }                                                                                      \
    } while (0)

const ov::PartialShape kFc{32, 16};

layout f16_lay() {
    return layout{kFc, data_types::f16, format::bfyx};
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

struct fc_graph {
    topology t;
    ov::intel_gpu::ImplForcingMap forced;
    std::vector<std::string> outs;
    std::vector<std::string> inputs;
    std::vector<std::string> writers;
};

void add_fc(fc_graph& g, engine& eng, const std::string& id, const std::string& in, int variant, impl_types t) {
    g.t.add(data(id + "_w", make_fc_weights(eng, 16, 16, variant)));
    g.t.add(fully_connected(id, input_info(in), id + "_w"));
    g.forced[id] = ov::intel_gpu::ImplementationDesc{format::bfyx, "", t};
}

// x_proj -> f1 -> f2 -> p(onednn FC) ; res = sum(p, x_proj): one writer, root x_proj.
fc_graph chain_graph(engine& eng) {
    fc_graph g;
    g.t.add(input_layout("in", f16_lay()));
    g.inputs = {"in"};
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

// O -> V (no-op strided_slice view); W1 = FC_onednn + sum(V); W2 = FC_onednn + sum(W1); Zp = prod(O, S2) reads O after W1 wrote: 1a downgrades W1,
// so W1 becomes the root of the {W1 -> W2} chain (origin downgraded_by_1a).
fc_graph chain_mid_graph(engine& eng) {
    fc_graph g;
    g.t.add(input_layout("in3", f16_lay()));
    g.t.add(input_layout("in1", f16_lay()));
    g.inputs = {"in3", "in1"};
    add_fc(g, eng, "O", "in3", 1, impl_types::ocl);
    g.t.add(strided_slice("V", input_info("O"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
    add_fc(g, eng, "W1", "in1", 0, impl_types::onednn);
    g.t.add(eltwise("S1", {input_info("W1"), input_info("V")}, eltwise_mode::sum));
    g.t.add(activation("a1", input_info("S1"), activation_func::abs));
    add_fc(g, eng, "W2", "a1", 4, impl_types::onednn);
    g.t.add(eltwise("S2", {input_info("W2"), input_info("S1")}, eltwise_mode::sum));
    g.t.add(eltwise("Zp", {input_info("O"), input_info("S2")}, eltwise_mode::prod));
    add_fc(g, eng, "Zb", "S2", 5, impl_types::ocl);
    g.t.add(reorder("out_Zb", input_info("Zb"), format::bfyx, data_types::f32));
    add_fc(g, eng, "Z", "Zp", 2, impl_types::ocl);
    g.t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.outs = {"out_Z", "out_Zb"};
    g.writers = {"W1", "W2"};
    return g;
}

ExecutionConfig real_cfg(engine& eng, const fc_graph& g, bool sum_switch) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(g.forced));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    config.set_property(ov::intel_gpu::enable_sum_root_pool(sum_switch));
#ifdef GPU_DEBUG_CONFIG
    config.set_user_property({ov::intel_gpu::dump_alloc_trace(1)});
#endif
    return config;
}

program::ptr build_real(engine& eng, const fc_graph& g, bool sum_switch) {
    return program::build_program(eng, g.t, real_cfg(eng, g, sum_switch), /*is_internal=*/false, /*no_optimizations=*/false);
}

// the old inline logic of basic_memory_dependencies (before P1-C-0), kept here as the reference of C0-2
struct inline_marks {
    program_node* eltw = nullptr;
    program_node* root = nullptr;
    program_node* owner = nullptr;
    bool is_sum = false;
};

inline_marks inline_walk(program_node& node) {
    inline_marks r;
    if (node.get_preferred_impl_type() != impl_types::onednn) {
        return r;
    }
    const int32_t reused_idx = fusing::get_reused_eltwmem_idx(node);
    if (reused_idx == -1) {
        return r;
    }
    for (const auto& fused_op : node.get_fused_primitives()) {
        if (fused_op.is_type<eltwise>() && fused_op.deps.size() == 1 && fused_op.has_outer_dep() && fused_op.outer_dep_start_idx == reused_idx &&
            fused_op.typed_desc<eltwise>()->mode == eltwise_mode::sum) {
            r.is_sum = true;
            break;
        }
    }
    if (!r.is_sum) {
        return r;
    }
    r.eltw = &node.get_dependency(static_cast<size_t>(reused_idx));
    auto* root = r.eltw;
    while (root->can_be_optimized() && !root->get_dependencies().empty()) {
        root = &root->get_dependency(0);
    }
    r.root = root;
    r.owner = const_cast<program_node*>(
        program_helpers::resolve_buffer_owner(*r.eltw, program_helpers::view_resolution::settled_only, false,
                                             node.get_dependency_with_port(static_cast<size_t>(reused_idx)).second)
            .owner);
    return r;
}

size_t count_formed_writers(program& p, const fc_graph& g) {
    size_t n = 0;
    for (const auto& w : g.writers) {
        n += (p.has_node(w) && fusing::sum_writer_root_marks(p.get_node(w)).has_value()) ? 1 : 0;
    }
    return n;
}

std::map<std::string, std::vector<float>> run_real(network& net, engine& eng, const fc_graph& g) {
    int variant = 1;
    for (const auto& in : g.inputs) {
        net.set_input_data(in, make_f16(eng, kFc, variant++));
    }
    auto outs = net.execute();
    std::map<std::string, std::vector<float>> r;
    for (const auto& id : g.outs) {
        auto mem = outs.at(id).get_memory();
        mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
        r[id] = std::vector<float>(lock.begin(), lock.end());
    }
    return r;
}

}  // namespace

TEST(sum_root_pool_census, C0_2_inline_walk_equals_helper_on_real_graphs) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    size_t formed_total = 0;
    for (auto make : {chain_graph, chain_mid_graph}) {
        auto g = make(eng);
        auto prog = build_real(eng, g, false);
        for (auto* n : prog->get_processing_order()) {
            const auto ref = inline_walk(*n);
            const auto got = fusing::sum_writer_root_marks(*n);
            ASSERT_EQ(got.has_value(), ref.is_sum) << n->id();
            if (!ref.is_sum) {
                continue;
            }
            ++formed_total;
            EXPECT_EQ(got->eltw_dep, ref.eltw) << n->id();
            EXPECT_EQ(got->legacy_root, ref.root) << n->id();
            EXPECT_EQ(got->owner, ref.owner) << n->id();
            EXPECT_EQ(got->reused_idx, fusing::get_reused_eltwmem_idx(*n)) << n->id();
        }
    }
    if (formed_total == 0) {
        GTEST_SKIP() << "construction not formed: no in-place sum formed in either graph";
    }
}

TEST(sum_root_pool_census, C0_10_census_on_a_real_single_writer_chain) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = chain_graph(eng);
    auto prog = build_real(eng, g, true);
    if (count_formed_writers(*prog, g) != 1) {
        GTEST_SKIP() << "construction not formed: the sum of p did not form";
    }
    std::string why;
    ASSERT_TRUE(class_level::sum_root_pool_active(*prog, &why)) << why;
    const auto census = class_level::run_sum_census(*prog);
    std::cout << "[census] chain_graph: " << census.summary() << "\n";
    for (const auto& c : census.classes) {
        std::cout << "[census]   root=" << c.root_id << " members=" << c.members << " writers=" << c.writers << " views=" << c.views
                  << " readers=" << c.readers << " root_bytes=" << c.root_bytes << " origin=" << (c.downgraded_by_1a ? "downgraded_by_1a" : "natural")
                  << " mask=" << class_level::to_string(c.mask) << "\n";
    }
    EXPECT_EQ(census.candidates, 1u) << census.summary();
    EXPECT_EQ(census.no_edge_sum, 0u);
    EXPECT_EQ(census.downgraded_roots, 0u);
    EXPECT_EQ(census.view_owner_closures, 0u);
    ASSERT_EQ(census.classes.size(), 1u);
    EXPECT_EQ(census.classes[0].root_id, "x_proj");
    EXPECT_EQ(census.classes[0].writers, 1u);
    EXPECT_EQ(census.classes[0].root_bytes, 32u * 16u * 2u);
    EXPECT_EQ(census.classes[0].mask & class_level::rj_sum_provenance, 0u) << "a clean single owner chain must pass L1S-3p";
    EXPECT_EQ(census.classes[0].mask & class_level::rj_program_level, 0u);
}

TEST(sum_root_pool_census, C0_11_census_downgraded_root_is_counted_as_downgraded_by_1a) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = chain_mid_graph(eng);
    auto prog = build_real(eng, g, true);
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    if (rep->entries.size() != 1 || rep->entries[0].writer_id != "W1" || count_formed_writers(*prog, g) != 1) {
        GTEST_SKIP() << "construction not formed: expected W1 downgraded by 1a and W2 kept; report: " << rep->summary();
    }
    const auto census = class_level::run_sum_census(*prog);
    std::cout << "[census] chain_mid_graph: " << census.summary() << "\n";
    EXPECT_EQ(census.downgraded_roots, 1u) << census.summary();
    EXPECT_EQ(census.no_edge_sum, 0u) << census.summary();
    bool found = false;
    for (const auto& c : census.classes) {
        if (c.root_id == "W1") {
            found = true;
            EXPECT_TRUE(c.downgraded_by_1a);
            EXPECT_EQ(c.writers, 1u) << "W2 is the only writer of the {W1 -> W2} class";
            EXPECT_EQ(c.mask & class_level::rj_sum_provenance, 0u) << "W2 marks (W1, W1): l == o, l is the root";
        }
    }
    EXPECT_TRUE(found) << "the class rooted at the downgraded writer W1 must be a census candidate: " << census.summary();
}

// C0-7b: the census-only switch changes nothing: flags, restrictions, pool fields, outputs and (GPU_DEBUG) the allocation trace.
TEST(sum_root_pool_census, C0_7b_census_only_switch_on_is_bit_identical_on_real_sum_graphs) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    for (auto make : {chain_graph, chain_mid_graph}) {
        auto g = make(eng);
        auto on = build_real(eng, g, true);
        auto off = build_real(eng, g, false);
        if (count_formed_writers(*on, g) == 0) {
            GTEST_SKIP() << "construction not formed: no in-place sum formed";
        }
        expect_same_state(*on, *off, "real graph switch on vs off");
        network net_on(on, 0), net_off(off, 0);
        const auto out_on = run_real(net_on, eng, g);
        const auto out_off = run_real(net_off, eng, g);
        EXPECT_EQ(out_on, out_off) << "outputs must be bit-identical";
#ifdef GPU_DEBUG_CONFIG
        const auto a = net_on.get_alloc_trace_summary();
        const auto b = net_off.get_alloc_trace_summary();
        EXPECT_EQ(a.total_count, b.total_count);
        EXPECT_EQ(a.at(alloc_category::pooled).count, b.at(alloc_category::pooled).count);
        const auto sa = net_on.get_held_bytes_snapshot();
        const auto sb = net_off.get_held_bytes_snapshot();
        EXPECT_EQ(sa.pooled.bytes, sb.pooled.bytes);
        EXPECT_EQ(sa.off_pool_bytes(), sb.off_pool_bytes());
        EXPECT_EQ(sa.total_bytes, sb.total_bytes);
#endif
    }
}

#endif  // ENABLE_ONEDNN_FOR_GPU
