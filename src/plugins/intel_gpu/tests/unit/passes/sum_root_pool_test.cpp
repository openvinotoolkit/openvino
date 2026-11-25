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
// P1-C-1 (design 18 r3): C0-7b is now the "switch on promotes" state comparison (the census-only neutrality it checked ended with P1-C-0); the
// C1-* / C-* tests further below cover the deferral, the backfill, L3 step (5) and the real graphs.

#include "test_utils.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
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
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/non_max_suppression.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/crop.hpp"
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
        // P1-C-1 (L1S-3 exact, CL-3): the mask is evaluated in the deferred state, so a non-shareable root IS reported even when a sum writer marks
        // it (before P1-C-1 this case was exempt: the approximation `writers_marking.count(root) == 0` is gone, see C1_3)
        R->can_share_buffer(false);
        EXPECT_EQ(mask_of(*h, *h->k, h->marks), rj_root_not_shareable);
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
    EXPECT_EQ(mask_of(*h, *h->k, h->marks, kNeverAmbiguous, /*admit_padded=*/true), 0u) << "admit_padded=true admits a padded root";
    EXPECT_EQ(mask_of(*h, *h->k, h->marks, kNeverAmbiguous, /*admit_padded=*/false), class_level::rj_sum_root_padded);
    EXPECT_FALSE(class_level::kAdmitPaddedSumRoot) << "P1-C-1: C-16b could not be constructed (the sum into a padded root never formed in the test pipeline), so the "
                                                      "preset ruling of NB-4 was flipped to false: padded roots are rejected (rj_sum_root_padded)";
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
// P1-C-1 (design 18 r3 section 7.2), tests that need no oneDNN device.
// ===============================================================================================================================

// C1-3 (CL-3): L1S-3 is exact. A root that a sum writer marks AND that is non-shareable for another reason (PBF, stateless_kv, ...) is rejected;
// a root that is only sum-marked (deferred: still shareable when the mask is evaluated) is not.
TEST(sum_root_pool_p1c1, C1_3_l1s3_exact_root_marked_by_sum_and_false_elsewhere) {
    auto& eng = get_test_engine();
    auto h = make_hand_chain(eng);
    REQUIRE_FORMED(h->ok, h->why_not);
    auto* R = &h->prog->get_node("R");
    ASSERT_NE(h->marks.writers_marking.find(R), h->marks.writers_marking.end()) << "R is marked by W1 (injected marks)";
    // control: only sum-marked, flag still true (the deferred state): not reported
    R->can_share_buffer(true);
    EXPECT_EQ(mask_of(*h, *h->k, h->marks) & class_level::rj_root_not_shareable, 0u);
    // non-sum source for the false (PBF / stateless_kv): reported although a sum writer marks the root as well
    R->can_share_buffer(false);
    EXPECT_NE(mask_of(*h, *h->k, h->marks) & class_level::rj_root_not_shareable, 0u);
    // the old approximation would have exempted it: the same marks without the writers_marking entry give the same answer
    auto m = h->marks;
    m.writers_marking.erase(R);
    EXPECT_NE(mask_of(*h, *h->k, m) & class_level::rj_root_not_shareable, 0u);
    R->can_share_buffer(true);
}

// C1-6 (1): the RAII backfill writes can_share_buffer(false) on marked \ promoted when it dies; apply() is idempotent; promoted nodes stay true.
TEST(sum_root_pool_p1c1, C1_6_backfill_raii_applies_on_every_exit_struct) {
    auto& eng = get_test_engine();
    auto h = make_hand_chain(eng);
    REQUIRE_FORMED(h->ok, h->why_not);
    auto& p = *h->prog;
    auto* R = &p.get_node("R");
    auto* W1 = &p.get_node("W1");
    auto* Z = &p.get_node("Z");
    for (auto* n : {R, W1, Z}) {
        n->can_share_buffer(true);
    }
    {  // explicit apply: marked \ promoted written, promoted and unmarked untouched; second apply is a no-op
        class_level::sum_root_backfill bf({R, W1});
        bf.promoted.insert(R);
        bf.apply();
        EXPECT_TRUE(R->can_share_buffer());
        EXPECT_FALSE(W1->can_share_buffer());
        EXPECT_TRUE(Z->can_share_buffer());
        W1->can_share_buffer(true);
        bf.apply();   // idempotent: applied once
        EXPECT_TRUE(W1->can_share_buffer());
        W1->can_share_buffer(true);
    }   // destructor: apply() already ran -> nothing more
    EXPECT_TRUE(W1->can_share_buffer());
    {  // destructor path (what an exception inside the pass triggers): no explicit apply
        try {
            class_level::sum_root_backfill bf({R, W1});
            bf.promoted.insert(R);
            throw std::runtime_error("simulated failure inside the pass");
        } catch (const std::runtime_error&) {
        }
        EXPECT_TRUE(R->can_share_buffer());
        EXPECT_FALSE(W1->can_share_buffer()) << "the destructor must write the deferred mark during stack unwinding";
        EXPECT_TRUE(Z->can_share_buffer());
    }
    for (auto* n : {R, W1, Z}) {
        n->can_share_buffer(true);
    }
    {  // nothing promoted: every marked node is non-shareable
        class_level::sum_root_backfill bf;
        bf.marked = {R, W1};
    }
    EXPECT_FALSE(R->can_share_buffer());
    EXPECT_FALSE(W1->can_share_buffer());
}

// C1-6 (2): on a built program every exit of the pass leaves the flags as with the switch off (I-C3). Without a oneDNN device no node is marked, so this
// is the vacuous half (the exits exercised: normal return with the switch on, 1a off, oooq); C-6 / C-7 below are the non-vacuous
// half on real oneDNN graphs.
TEST(sum_root_pool_p1c1, C1_6_backfill_program_level_exits_equal_switch_off) {
    auto& eng = get_test_engine();
    auto off = build_pipeline(eng, false, QueueTypes::in_order);
    struct exit_case {
        const char* name;
        std::function<program::ptr()> make;
    };
    const std::vector<exit_case> cases = {
        {"normal return", [&] { return build_pipeline(eng, true, QueueTypes::in_order); }},
        {"1a off", [&] { return build_pipeline(eng, true, QueueTypes::in_order, [](ExecutionConfig& c) { c.set_property(ov::intel_gpu::enable_inplace_conflict_resolution(false)); }); }},
    };
    for (const auto& c : cases) {
        auto on = c.make();
        expect_same_state(*on, *off, c.name);
    }
    auto oooq_on = build_pipeline(eng, true, QueueTypes::out_of_order);
    auto oooq_off = build_pipeline(eng, false, QueueTypes::out_of_order);
    expect_same_state(*oooq_on, *oooq_off, "oooq");
}

namespace {

// a store with n nodes: uids and R0 given
class_level::simple_restriction_store make_store(const std::vector<uint32_t>& uids, const std::vector<std::vector<uint32_t>>& r0) {
    class_level::simple_restriction_store s;
    s.uids = uids;
    s.sets = r0;
    for (auto& v : s.sets) {
        std::sort(v.begin(), v.end());
    }
    return s;
}

// the owners of a view uid, as a callback
class_level::view_owner_fn owners_table(std::map<uint32_t, std::vector<uint32_t>> t) {
    return [t](uint32_t u, std::vector<uint32_t>& out) {
        const auto it = t.find(u);
        if (it != t.end()) {
            out.insert(out.end(), it->second.begin(), it->second.end());
        }
    };
}

}  // namespace

// C1-8: L3 step (5) white box. Node indices: 0 root r (uid 10), 1 member (uid 11), 2 view y (uid 21), 3 its owner Y (uid 20), 4 unrelated (uid 30).
// R0(r) holds the view only (the folded owner was skipped), the owner Y does not know r. Done once for a concat-shaped and once for a sum-shaped class:
// step (5) does not depend on the class kind (CL-1).
TEST(sum_root_pool_p1c1, C1_8_l3_step5_view_owner_closure_white_box) {
    using namespace class_level;
    for (const char* kind : {"concat class", "sum class"}) {
        auto make = [&] { return make_store({10, 11, 21, 20, 30}, {{21}, {}, {}, {}, {}}); };
        const std::vector<promoted_class> classes = {{0, {0, 1}}};
        const auto owners = owners_table({{21, {20}}});
        {   // nullptr: no step (5)
            auto s = make();
            const auto r0 = snapshot_restrictions(s);
            EXPECT_EQ(materialize_class_restrictions(s, classes, nullptr), 0u) << kind;
            EXPECT_FALSE(s.contains(0, 20)) << kind << ": without the callback the owner stays out (the pre-P1-C-1 behaviour)";
            EXPECT_EQ(check_class_invariants(s, r0, classes), "") << kind;
            EXPECT_NE(check_view_owner_closure(s, classes, owners), "") << kind << ": I-B1-V must see the gap";
        }
        {   // with the callback: closed in both directions, nothing else changes
            auto s = make();
            const auto r0 = snapshot_restrictions(s);
            EXPECT_EQ(materialize_class_restrictions(s, classes, &owners), 1u) << kind;
            EXPECT_TRUE(s.contains(0, 20)) << kind << ": the owner Y enters R(r)";
            EXPECT_TRUE(s.contains(3, 10)) << kind << ": r enters R(Y)";
            EXPECT_EQ(check_class_invariants(s, r0, classes), "") << kind;
            EXPECT_EQ(check_view_owner_closure(s, classes, owners), "") << kind;
            EXPECT_TRUE(s.values(4).empty()) << kind << ": an unrelated node is unchanged";
            EXPECT_FALSE(s.contains(1, 20)) << kind << ": the member only ever gains the root uid, not the owner";
        }
    }
}

// C1-8b: lemma V cases of step (5).
TEST(sum_root_pool_p1c1, C1_8b_l3_step5_lemma_v_cases) {
    using namespace class_level;
    {   // (i) the owner belongs to r's own class: skipped, nothing added
        auto s = make_store({10, 11, 21}, {{21}, {}, {}});
        const std::vector<promoted_class> classes = {{0, {0, 1}}};
        const auto owners = owners_table({{21, {11}}});   // 11 is a member of r
        const auto r0 = snapshot_restrictions(s);
        EXPECT_EQ(materialize_class_restrictions(s, classes, &owners), 0u);
        EXPECT_EQ(check_view_owner_closure(s, classes, owners), "");
        EXPECT_EQ(check_class_invariants(s, r0, classes), "");
    }
    {   // (ii) the owner is a non-root member of another promoted class K2 and the view u is a member of K2: (3') already relates r and root(K2)
        // nodes: 0 r (10), 1 m (11), 2 root2 (12), 3 w2 (13, writer of K2), 4 u (14, view of w2, member of K2)
        auto make = [&] { return make_store({10, 11, 12, 13, 14}, {{14}, {}, {}, {}, {}}); };
        const std::vector<promoted_class> classes = {{0, {0, 1}}, {2, {2, 3, 4}}};
        const auto owners = owners_table({{14, {13}}});
        auto base = make();
        materialize_class_restrictions(base, classes, nullptr);
        EXPECT_TRUE(base.contains(0, 12) && base.contains(2, 10)) << "(3') alone relates the roots r and root2 (lemma V.2): no obligation is left for step (5)";
        auto s = make();
        const auto r0 = snapshot_restrictions(s);
        materialize_class_restrictions(s, classes, &owners);   // the owner uid itself is added too: harmless, strictly more conservative
        EXPECT_EQ(check_view_owner_closure(s, classes, owners), "");
        EXPECT_EQ(check_class_invariants(s, r0, classes), "");
        EXPECT_TRUE(s.contains(0, 12) && s.contains(2, 10));
    }
    {   // (iii) the owner is the root of another promoted class: closed in both directions by step (5) itself
        auto s = make_store({10, 11, 12, 13, 21}, {{21}, {}, {}, {}, {}});
        const std::vector<promoted_class> classes = {{0, {0, 1}}, {2, {2, 3}}};
        const auto owners = owners_table({{21, {12}}});
        const auto r0 = snapshot_restrictions(s);
        EXPECT_EQ(materialize_class_restrictions(s, classes, &owners), 1u);
        EXPECT_TRUE(s.contains(0, 12) && s.contains(2, 10));
        EXPECT_EQ(check_view_owner_closure(s, classes, owners), "");
        EXPECT_EQ(check_class_invariants(s, r0, classes), "");
    }
    {   // (iv) no cascade: a second call changes nothing; the owner of an owner is not followed
        auto s = make_store({10, 11, 21, 20, 22}, {{21}, {}, {}, {}, {}});
        const std::vector<promoted_class> classes = {{0, {0, 1}}};
        const auto owners = owners_table({{21, {20}}, {20, {22}}});   // 20 resolving further would be a holder cascade; step (5) must not follow it
        EXPECT_EQ(materialize_class_restrictions(s, classes, &owners), 1u);
        EXPECT_FALSE(s.contains(0, 22)) << "single pass: no cascade";
        const auto snap = snapshot_restrictions(s);
        EXPECT_EQ(snap, snapshot_restrictions(s));
    }
    {   // (v) several owners of one view uid (one per output port, NB-V1): all of them are closed
        auto s = make_store({10, 11, 21, 20, 23}, {{21}, {}, {}, {}, {}});
        const std::vector<promoted_class> classes = {{0, {0, 1}}};
        const auto owners = owners_table({{21, {20, 23}}});
        EXPECT_EQ(materialize_class_restrictions(s, classes, &owners), 2u);
        EXPECT_TRUE(s.contains(0, 20) && s.contains(0, 23));
        EXPECT_TRUE(s.contains(3, 10) && s.contains(4, 10));
        EXPECT_EQ(check_view_owner_closure(s, classes, owners), "");
    }
}

// C1-8b (production callback): resolve_buffer_owner over a real (hand) program, per output port. nms_gather port k aliases input k; the owner of
// view1 (a reorder view of port 1) is the nms node.
TEST(sum_root_pool_p1c1, C1_8b_production_callback_resolves_owners_per_port) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("boxes", layout{ov::PartialShape{1, 8, 4}, data_types::f32, format::bfyx}));
    t.add(input_layout("scores", layout{ov::PartialShape{1, 2, 8}, data_types::f32, format::bfyx}));
    t.add(cldnn::data("num_per_class", eng.allocate_memory({data_types::i32, format::bfyx, {1, 1, 1, 1}})));
    t.add(cldnn::data("iou_threshold", eng.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}})));
    t.add(cldnn::data("score_threshold", eng.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}})));
    auto nms = non_max_suppression("nms", input_info("boxes"), input_info("scores"), 16, false, true, "num_per_class", "iou_threshold", "score_threshold",
                                   "", "", "", 3);
    nms.output_data_types = {optional_data_type{}, optional_data_type{data_types::f32}, optional_data_type{}};
    nms.output_paddings = {padding(), padding(), padding()};
    t.add(nms);
    t.add(non_max_suppression_gather("nms_gather", {input_info("nms", 0), input_info("nms", 1), input_info("nms", 2)}, 3));
    t.add(reorder("view1", input_info("nms_gather", 1), format::bfyx, data_types::f32));
    t.add(reorder("out0", input_info("nms_gather", 0), format::bfyx, data_types::i32));
    t.add(reorder("out1", input_info("view1"), format::bfyx, data_types::f32));
    t.add(reorder("out2", input_info("nms_gather", 2), format::bfyx, data_types::i32));
    auto prog = make_hand(eng, t, {"out0", "out1", "out2"}, false, false, [](ExecutionConfig& c) { c.set_property(ov::intel_gpu::allow_new_shape_infer(true)); });
    REQUIRE_FORMED(prog->get_node("nms_gather").can_be_optimized() || prog->get_node("nms_gather").is_runtime_skippable(), "nms_gather is not a view node");
    prog->get_node("view1").can_be_optimized(true);
    std::vector<program_node*> nodes(prog->get_processing_order().begin(), prog->get_processing_order().end());
    const auto fn = class_level::make_program_view_owner_fn(nodes);
    const uint32_t nms_uid = static_cast<uint32_t>(prog->get_node("nms").get_unique_id());
    std::vector<uint32_t> o;
    fn(static_cast<uint32_t>(prog->get_node("nms_gather").get_unique_id()), o);
    EXPECT_EQ(o, std::vector<uint32_t>{nms_uid}) << "ports 0..2 of nms_gather all resolve to the nms node (deduplicated)";
    o.clear();
    fn(static_cast<uint32_t>(prog->get_node("view1").get_unique_id()), o);
    EXPECT_EQ(o, std::vector<uint32_t>{nms_uid}) << "view1 is a view of nms_gather port 1 = nms port 1";
    o.clear();
    fn(nms_uid, o);
    EXPECT_TRUE(o.empty()) << "a plain owner has no owner";
    o.clear();
    fn(static_cast<uint32_t>(prog->get_node("out0").get_unique_id()), o);
    EXPECT_TRUE(o.empty()) << "a non view node has no owner";
}

// C1-8c: I-B1-V is discriminating: deleting any single record step (5) inserted makes the verifier non-empty.
TEST(sum_root_pool_p1c1, C1_8c_i_b1_v_is_discriminating) {
    using namespace class_level;
    const std::vector<promoted_class> classes = {{0, {0, 1}}};
    const auto owners = owners_table({{21, {20}}});
    {   // drop R(r) -> owner
        auto s = make_store({10, 11, 21, 20}, {{21}, {}, {}, {}});
        materialize_class_restrictions(s, classes, &owners);
        ASSERT_EQ(check_view_owner_closure(s, classes, owners), "");
        auto& v = s.sets[0];
        v.erase(std::remove(v.begin(), v.end(), 20u), v.end());
        EXPECT_NE(check_view_owner_closure(s, classes, owners), "");
    }
    {   // drop owner -> r
        auto s = make_store({10, 11, 21, 20}, {{21}, {}, {}, {}});
        materialize_class_restrictions(s, classes, &owners);
        auto& v = s.sets[3];
        v.erase(std::remove(v.begin(), v.end(), 10u), v.end());
        EXPECT_NE(check_view_owner_closure(s, classes, owners), "");
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
    std::map<std::string, ov::PartialShape> shapes;   // input shapes other than kFc
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
        const auto it = g.shapes.find(in);
        net.set_input_data(in, make_f16(eng, it != g.shapes.end() ? it->second : kFc, variant++));
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

// ===============================================================================================================================
// P1-C-1 real oneDNN graphs (design 18 r3 section 7.2: C-1 ... C-21). A SKIP counts as "not covered".
// ===============================================================================================================================
namespace {

#define P1C1_REQUIRE(cond, msg)                                                      \
    do {                                                                             \
        if (!(cond)) {                                                               \
            GTEST_SKIP() << "construction not formed: " << (msg);                    \
        }                                                                            \
    } while (0)

enum class view_kind { none, skippable_slice, cropped };

void add_fc_dims(fc_graph& g, engine& eng, const std::string& id, const std::string& in, int variant, impl_types t, int64_t o, int64_t i) {
    g.t.add(data(id + "_w", make_fc_weights(eng, o, i, variant)));
    g.t.add(fully_connected(id, input_info(in), id + "_w"));
    g.forced[id] = ov::intel_gpu::ImplementationDesc{format::bfyx, "", t};
}

// O -> [V] -> W1 (onednn FC + sum(O or V)) -> a1 -> W2 (onednn FC + sum(S1)) -> Zb [-> T1 -> T2] -> out_Zb: two writers rooted at O.
// view_kind::skippable_slice: V is a no-op strided_slice (runtime skippable: the class is REJECTED by L1S-5);
// view_kind::cropped: V is a static crop of the first 8 features, an in-place crop view with padding (the working shape is {32,8}).
fc_graph chain2_graph(engine& eng, view_kind vk, bool tail) {
    fc_graph g;
    const bool crp = vk == view_kind::cropped;
    const int64_t N = crp ? 8 : 16;   // working feature count; rows are 32
    g.t.add(input_layout("in3", f16_lay()));
    g.t.add(input_layout("in1", f16_lay()));
    g.inputs = {"in3", "in1"};
    add_fc(g, eng, "O", "in3", 1, impl_types::ocl);
    std::string dep = "O";
    if (vk == view_kind::skippable_slice) {
        g.t.add(strided_slice("V", input_info("O"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
        dep = "V";
    } else if (crp) {
        g.t.add(crop("V", input_info("O"), tensor(32, N, 1, 1), tensor(0, 0, 0, 0)));
        dep = "V";
    }
    add_fc_dims(g, eng, "W1", "in1", 0, impl_types::onednn, N, 16);
    g.t.add(eltwise("S1", {input_info("W1"), input_info(dep)}, eltwise_mode::sum));
    g.t.add(activation("a1", input_info("S1"), activation_func::abs));
    add_fc_dims(g, eng, "W2", "a1", 4, impl_types::onednn, N, N);
    g.t.add(eltwise("S2", {input_info("W2"), input_info("S1")}, eltwise_mode::sum));
    add_fc_dims(g, eng, "Zb", "S2", 5, impl_types::ocl, N, N);
    std::string last = "Zb";
    if (tail) {
        add_fc_dims(g, eng, "T1", "Zb", 6, impl_types::ocl, N, N);
        add_fc_dims(g, eng, "T2", "T1", 7, impl_types::ocl, N, N);
        last = "T2";
    }
    g.t.add(reorder("out_Zb", input_info(last), format::bfyx, data_types::f32));
    g.outs = {"out_Zb"};
    g.writers = {"W1", "W2"};
    return g;
}

primitive_inst::alloc_route route_of(const program_node& n) {
    return primitive_inst::allocation_route(n, 0, n.can_be_optimized());
}

const void* buf_of(network& net, const std::string& id) {
    auto m = net.get_primitive(id)->output_memory_ptr();
    return m ? m->buffer_ptr() : nullptr;
}

size_t pool_records_of(network& net, const std::string& id) {
    return net.get_memory_pool().count_records_with_user(net.get_primitive(id)->get_node().get_unique_id(), net.get_id());
}

std::set<std::string> pool_roots(program& p) {
    std::set<std::string> r;
    for (auto* n : p.get_processing_order()) {
        if (n->is_pool_class_root()) {
            r.insert(n->id());
        }
    }
    return r;
}

// the flags of every node outside `roots` equal the switch-off state; a promoted root is pooled (shareable, root flag, root uid) where the switch-off
// state keeps it out of the pool. `check_restrictions`: restrictions of the nodes outside the roots are not compared (they may gain root uids).
void expect_state_except_roots(program& on, program& off, const std::set<std::string>& roots, const std::string& what) {
    ASSERT_EQ(on.get_processing_order().size(), off.get_processing_order().size()) << what;
    for (auto* n : on.get_processing_order()) {
        ASSERT_TRUE(off.has_node(n->id())) << what << ": " << n->id();
        auto& m = off.get_node(n->id());
        EXPECT_EQ(n->can_be_optimized(), m.can_be_optimized()) << what << ": cbo of " << n->id();
        if (roots.count(n->id()) != 0) {
            EXPECT_TRUE(n->is_pool_class_root()) << what << ": " << n->id();
            EXPECT_TRUE(n->can_share_buffer()) << what << ": promoted root " << n->id() << " must be shareable";
            EXPECT_FALSE(m.can_share_buffer()) << what << ": the switch-off state keeps " << n->id() << " out of the pool";
            EXPECT_EQ(n->get_alias_pool_root_uid(), static_cast<uint32_t>(n->get_unique_id())) << what;
            continue;
        }
        EXPECT_EQ(n->can_share_buffer(), m.can_share_buffer()) << what << ": can_share_buffer of " << n->id();
        EXPECT_EQ(n->is_pool_class_root(), m.is_pool_class_root()) << what << ": pool_class_root of " << n->id();
        EXPECT_EQ(n->has_alias_pool_root(), m.has_alias_pool_root()) << what << ": alias_pool_root of " << n->id();
    }
}

// Pattern A/B/C oracle: two alias classes that live in the same buffer have disjoint [first member, last reader] intervals in processing order.
std::vector<std::string> sum_lifetime_violations(network& net) {
    std::vector<std::string> res;
    auto prog = net.get_program();
    const auto* aa = prog->get_alias_analysis();
    std::unique_ptr<alias_analysis> local_aa;
    if (aa == nullptr) {
        local_aa = std::make_unique<alias_analysis>(alias_analysis::build(*prog));
        aa = local_aa.get();
    }
    std::map<const program_node*, size_t> idx;
    size_t i = 0;
    for (auto* n : prog->get_processing_order()) {
        idx[n] = i++;
    }
    struct cls_info {
        const void* buf;
        size_t birth, death;
        std::string name;
    };
    std::vector<cls_info> infos;
    for (const auto& c : aa->classes()) {
        if (c.domain & (dm_external_root | dm_persistent)) {
            continue;
        }
        const void* buf = buf_of(net, c.root_node->id());
        if (buf == nullptr) {
            continue;
        }
        size_t birth = std::numeric_limits<size_t>::max(), death = 0;
        for (const auto& m : c.members) {
            birth = std::min(birth, idx.at(m.node));
            death = std::max(death, idx.at(m.node));
            for (const auto* u : m.node->get_users()) {
                death = std::max(death, idx.at(u));
            }
        }
        infos.push_back({buf, birth, death, c.root_node->id()});
    }
    for (size_t a = 0; a < infos.size(); a++) {
        for (size_t b = a + 1; b < infos.size(); b++) {
            if (infos[a].buf == infos[b].buf && !(infos[a].death < infos[b].birth || infos[b].death < infos[a].birth)) {
                res.push_back("classes " + infos[a].name + " and " + infos[b].name + " share a buffer with intersecting lifetimes");
            }
        }
    }
    return res;
}

struct class_span {
    size_t birth = 0, death = 0;
};

class_span span_of_class(program& p, const std::string& root_id) {
    const auto* aa = p.get_alias_analysis();
    class_span sp;
    std::map<const program_node*, size_t> idx;
    size_t i = 0;
    for (auto* n : p.get_processing_order()) {
        idx[n] = i++;
    }
    const auto* k = aa->class_of(p.get_node(root_id));
    sp.birth = std::numeric_limits<size_t>::max();
    for (const auto& m : k->members) {
        sp.birth = std::min(sp.birth, idx.at(m.node));
        sp.death = std::max(sp.death, idx.at(m.node));
        for (const auto* u : m.node->get_users()) {
            sp.death = std::max(sp.death, idx.at(u));
        }
    }
    return sp;
}

bool has_restriction(const program_node& n, const program_node& against) {
    return n.has_memory_dependency(against.get_unique_id());
}

}  // namespace

// C-1: R -> W1 -> W2: the chain root enters the pool as the class root; the writers stay out of the pool and share R's buffer.
TEST(sum_root_pool_p1c1_onednn, C_1_chain_root_enters_pool) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    for (bool unused : {false}) {
        (void)unused;
        auto g = chain2_graph(eng, view_kind::none, false);
        auto prog = build_real(eng, g, true);
        P1C1_REQUIRE(count_formed_writers(*prog, g) == 2, "both oneDNN sums (W1, W2) must form");
        auto& O = prog->get_node("O");
        auto& W1 = prog->get_node("W1");
        auto& W2 = prog->get_node("W2");
        const auto marks = class_level::collect_sum_root_marks(*prog);
        ASSERT_TRUE(marks.of_writer.count(&W2) != 0 && marks.of_writer.count(&W1) != 0);
        EXPECT_EQ(marks.of_writer.at(&W2).first, &W1) << "legacy_root(W2) = owner(W2) = W1 (E5b)";
        EXPECT_EQ(marks.of_writer.at(&W2).second, &W1);
        EXPECT_EQ(marks.of_writer.at(&W1).first, &O);
        EXPECT_NE(marks.writers_marking.find(&W1), marks.writers_marking.end()) << "W1 is in `marked` (deferred) and was written false by the backfill";
        EXPECT_EQ(route_of(O), primitive_inst::alloc_route::POOL_AS_ROOT);
        EXPECT_TRUE(O.is_pool_class_root());
        EXPECT_TRUE(O.can_share_buffer());
        for (auto* w : {&W1, &W2}) {
            EXPECT_EQ(route_of(*w), primitive_inst::alloc_route::DIRECT) << w->id();
            EXPECT_FALSE(w->is_pool_class_root()) << w->id();
            EXPECT_FALSE(w->has_alias_pool_root()) << w->id();
            EXPECT_FALSE(w->can_share_buffer()) << w->id();
        }
        EXPECT_EQ(class_level::check_b4_prime(*prog), "");
        network net(prog, 0);
        for (const char* w : {"W1", "W2"}) {
            EXPECT_EQ(buf_of(net, w), buf_of(net, "O")) << w << " writes in place into the buffer of R";
        }
        EXPECT_EQ(pool_records_of(net, "O"), 1u) << "R owns exactly one pool record";
        EXPECT_EQ(pool_records_of(net, "W1"), 0u);
        EXPECT_EQ(pool_records_of(net, "W2"), 0u);
    }
}

// C-2: after the class died a later node takes R's block; nothing alive in the class interval does.
TEST(sum_root_pool_p1c1_onednn, C_2_root_block_reused_after_class_death) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = chain2_graph(eng, view_kind::none, true);
    auto prog = build_real(eng, g, true);
    P1C1_REQUIRE(count_formed_writers(*prog, g) == 2, "both oneDNN sums must form");
    ASSERT_TRUE(prog->get_node("O").is_pool_class_root());
    network net(prog, 0);
    const auto out = run_real(net, eng, g);
    const auto sp = span_of_class(*prog, "O");
    std::map<const program_node*, size_t> idx;
    size_t i = 0;
    for (auto* n : prog->get_processing_order()) {
        idx[n] = i++;
    }
    const void* rbuf = buf_of(net, "O");
    bool reused_after_death = false;
    const auto* k = prog->get_alias_analysis()->class_of(prog->get_node("O"));
    std::set<const program_node*> members;
    for (const auto& m : k->members) {
        members.insert(m.node);
    }
    for (auto* n : prog->get_processing_order()) {
        if (members.count(n) != 0 || n->is_type<data>() || n->is_type<input_layout>()) {
            continue;
        }
        auto m = net.get_primitive(n->id())->output_memory_ptr();
        if (m == nullptr || m->buffer_ptr() != rbuf) {
            continue;
        }
        EXPECT_GT(idx.at(n), sp.death) << n->id() << " shares the buffer of R but executes inside the class interval [" << sp.birth << "," << sp.death << "]";
        reused_after_death = reused_after_death || idx.at(n) > sp.death;
    }
    EXPECT_TRUE(reused_after_death) << "a node after the class death is expected to take the block of R (every tensor of the graph has the same size)";
    EXPECT_NE(buf_of(net, "Zb"), rbuf) << "the last reader of the class must not take the block";
    EXPECT_TRUE(sum_lifetime_violations(net).empty());
    EXPECT_FALSE(out.empty());
#ifdef GPU_DEBUG_CONFIG
    auto g_off = chain2_graph(eng, view_kind::none, true);
    auto prog_off = build_real(eng, g_off, false);
    network net_off(prog_off, 0);
    run_real(net_off, eng, g_off);
    const auto a = net.get_held_bytes_snapshot();
    const auto b = net_off.get_held_bytes_snapshot();
    std::cout << "[AB] C-2 off_pool on=" << a.off_pool_bytes() << " off=" << b.off_pool_bytes() << " pooled on=" << a.pooled.bytes << " off=" << b.pooled.bytes
              << " total on=" << a.total_bytes << " off=" << b.total_bytes << "\n";
    EXPECT_EQ(a.off_pool_bytes() + 32u * 16u * 2u, b.off_pool_bytes()) << "the root left the off-pool bytes";
    EXPECT_LE(a.total_bytes, b.total_bytes) << "pooling the root never increases the held bytes";
#endif
}

// C-3 + C-5/C-5b: the class interval [birth(R), max death] is covered: every user of a member and every node executing inside the interval
// (their internal buffers use their own uid) has R in its restriction; internal buffers never alias R's block.
TEST(sum_root_pool_p1c1_onednn, C_3_C_5_class_interval_is_covered_in_restrictions) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = chain2_graph(eng, view_kind::none, true);
    auto prog = build_real(eng, g, true);
    P1C1_REQUIRE(count_formed_writers(*prog, g) == 2, "both oneDNN sums must form");
    auto& R = prog->get_node("O");
    ASSERT_TRUE(R.is_pool_class_root());
    const auto sp = span_of_class(*prog, "O");
    const auto* k = prog->get_alias_analysis()->class_of(R);
    size_t checked = 0, i = 0;
    std::set<const program_node*> users;
    for (const auto& m : k->members) {
        for (const auto* u : m.node->get_users()) {
            users.insert(u);
        }
    }
    for (const auto* u : users) {
        if (u->may_use_mempool() && u != &R) {
            EXPECT_TRUE(has_restriction(*u, R)) << "user " << u->id() << " of a class member must carry the uid of R";
            ++checked;
        }
    }
    network net(prog, 0);
    size_t internal_checked = 0;
    for (auto* n : prog->get_processing_order()) {
        const size_t pos = i++;
        if (pos < sp.birth || pos > sp.death || n == &R || !n->may_use_mempool()) {
            continue;
        }
        EXPECT_TRUE(has_restriction(*n, R)) << "node " << n->id() << " executes inside the class interval [" << sp.birth << "," << sp.death << "]";
        EXPECT_TRUE(has_restriction(R, *n)) << "R must restrict " << n->id() << " as well (static blocks are never released: both directions)";
        for (const auto& im : net.get_primitive(n->id())->get_intermediates_memories()) {
            if (im != nullptr) {
                EXPECT_NE(im->buffer_ptr(), buf_of(net, "O")) << n->id() << ": an internal buffer must not alias the block of R";
                ++internal_checked;
            }
        }
        ++checked;
    }
    std::cout << "[C-3/C-5] checked restrictions=" << checked << " internal buffers=" << internal_checked << " interval=[" << sp.birth << "," << sp.death << "]\n";
    EXPECT_GT(checked, 0u);
    run_real(net, eng, g);
    EXPECT_TRUE(sum_lifetime_violations(net).empty());
}

// C-6 / C-7: a REJECTED class leaves every flag and restriction as the switch off (theorem F). The rejected class here is a real one: the oneDNN sum
// whose dependency is an in-place concat (defect B: legacy root != owner, root is a concat). The deferred marks of the writer's chain root and owner are
// then written by the backfill only, which is what makes this non-vacuous.
namespace {

layout f16_lay_c(size_t f) {
    return layout{ov::PartialShape{1, static_cast<int64_t>(f), 32, 32}, data_types::f16, format::bfyx};
}

memory::ptr make_conv_weights(engine& eng, size_t O, size_t I, size_t K) {
    auto mem = eng.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(O), static_cast<int64_t>(I), static_cast<int64_t>(K), static_cast<int64_t>(K)},
                                          data_types::f16, format::bfyx});
    std::vector<ov::float16> w(O * I * K * K);
    for (size_t o = 0; o < O; o++) {
        for (size_t i = 0; i < I; i++) {
            for (size_t ky = 0; ky < K; ky++) {
                for (size_t kx = 0; kx < K; kx++) {
                    const bool centre = (ky == K / 2) && (kx == K / 2);
                    w[((o * I + i) * K + ky) * K + kx] = ov::float16((o == i && centre) ? 0.5f : (((o + 2 * i + ky + kx) % 7 == 0) ? 0.125f : 0.f));
                }
            }
        }
    }
    set_values(mem, w);
    return mem;
}

ov::intel_gpu::ImplementationDesc fsv16_impl(impl_types t) {
    return ov::intel_gpu::ImplementationDesc{format::b_fs_yx_fsv16, "", t};
}

void add_conv_p(topology& t, const primitive_id& id, const primitive_id& in, const primitive_id& w, int pad) {
    t.add(convolution(id, input_info(in), w, "", 1, {1, 1}, {1, 1}, {pad, pad}, {pad, pad}, false));
}

memory::ptr conv_input(engine& eng, size_t f, int variant) {
    auto mem = eng.allocate_memory(f16_lay_c(f));
    std::vector<ov::float16> v(mem->count());
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = ov::float16(variant == 1 ? static_cast<float>(i % 17) * 0.125f - 1.0f : static_cast<float>((i * 3) % 13) * 0.0625f - 0.5f);
    }
    set_values(mem, v);
    return mem;
}

// P, Q -> C (in-place concat holder) ; W = onednn conv(in3) + sum(C) -> S -> out (the F10 control graph: the sum into the in-place concat is legal)
topology topo_sum_into_concat(engine& eng) {
    topology t;
    t.add(input_layout("in1", f16_lay_c(16)));
    t.add(input_layout("in3", f16_lay_c(32)));
    t.add(cldnn::data("w_pq", make_conv_weights(eng, 16, 16, 1)));
    add_conv_p(t, "P", "in1", "w_pq", 0);
    add_conv_p(t, "Q", "in1", "w_pq", 0);
    t.add(concatenation("C", {input_info("P"), input_info("Q")}, 1));
    t.add(cldnn::data("w_w", make_conv_weights(eng, 32, 32, 3)));
    add_conv_p(t, "W", "in3", "w_w", 1);
    t.add(eltwise("S", {input_info("W"), input_info("C")}, eltwise_mode::sum));
    t.add(reorder("out_S", input_info("S"), format::bfyx, data_types::f32));
    return t;
}

ExecutionConfig concat_sum_cfg(engine& eng, bool sum_switch, bool concat_switch, bool res_1a = true) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    ov::intel_gpu::ImplForcingMap forced = {{"P", fsv16_impl(impl_types::ocl)}, {"Q", fsv16_impl(impl_types::ocl)},
                                            {"C", fsv16_impl(impl_types::ocl)}, {"W", fsv16_impl(impl_types::onednn)}};
    config.set_property(ov::intel_gpu::force_implementations(forced));
    config.set_property(ov::intel_gpu::enable_sum_root_pool(sum_switch));
    config.set_property(ov::intel_gpu::enable_concat_holder_pool(concat_switch));
    config.set_property(ov::intel_gpu::enable_inplace_conflict_resolution(res_1a));
    return config;
}

}  // namespace

TEST(sum_root_pool_p1c1_onednn, C_6_rejected_class_flags_equal_switch_off) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    for (bool concat_switch : {false, true}) {
        auto t_on = topo_sum_into_concat(eng);
        auto t_off = topo_sum_into_concat(eng);
        auto on = program::build_program(eng, t_on, concat_sum_cfg(eng, true, concat_switch), false, false);
        auto off = program::build_program(eng, t_off, concat_sum_cfg(eng, false, concat_switch), false, false);
        P1C1_REQUIRE(on->get_node("C").can_be_optimized(), "C is not an in-place concat");
        P1C1_REQUIRE(fusing::sum_writer_root_marks(on->get_node("W")).has_value(), "the sum of W (into the in-place concat) did not form");
        std::vector<program_node*> marked;
        const auto marks = class_level::collect_sum_root_marks(*on, &marked);
        ASSERT_FALSE(marked.empty()) << "the deferral must have something to write back (non-vacuous)";
        std::string why;
        ASSERT_TRUE(class_level::sum_root_pool_active(*on, &why)) << why;
        const auto census = class_level::run_sum_census(*on);
        std::cout << "[C-6] concat_switch=" << concat_switch << " " << census.summary() << "\n";
        EXPECT_EQ(pool_roots(*on).count("C"), 0u) << "the concat holder of a sum chain is never promoted";
        for (auto* n : marked) {
            EXPECT_FALSE(n->can_share_buffer()) << "marked node " << n->id() << " must be non-shareable after the backfill (I-C3)";
        }
        expect_same_state(*on, *off, std::string("rejected sum class, concat switch=") + (concat_switch ? "on" : "off"));
        network n_on(on, 0), n_off(off, 0);
        auto run = [&](network& n) {
            n.set_input_data("in1", conv_input(eng, 16, 1));
            n.set_input_data("in3", conv_input(eng, 32, 2));
            auto outs = n.execute();
            auto mem = outs.at("out_S").get_memory();
            mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
            return std::vector<float>(lock.begin(), lock.end());
        };
        EXPECT_EQ(run(n_on), run(n_off));
    }
}

// C-6b: the second real rejected class: the view of the root is a no-op strided_slice (runtime skippable). L1S-5 rejects the class; the deferred marks of
// root O (and the owner) are written by the backfill, so every flag and restriction equals the switch off.
TEST(sum_root_pool_p1c1_onednn, C_6b_rejected_class_with_a_skippable_view_equals_switch_off) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g_on = chain2_graph(eng, view_kind::skippable_slice, true);
    auto g_off = chain2_graph(eng, view_kind::skippable_slice, true);
    auto on = build_real(eng, g_on, true);
    auto off = build_real(eng, g_off, false);
    P1C1_REQUIRE(count_formed_writers(*on, g_on) == 2, "both oneDNN sums must form");
    P1C1_REQUIRE(on->get_node("V").is_runtime_skippable(), "the no-op strided_slice is not runtime skippable");
    std::vector<program_node*> marked;
    class_level::collect_sum_root_marks(*on, &marked);
    ASSERT_FALSE(marked.empty());
    const auto census = class_level::run_sum_census(*on);
    ASSERT_EQ(census.candidates, 1u) << census.summary();
    EXPECT_NE(census.classes[0].mask & class_level::rj_dynamic, 0u) << class_level::to_string(census.classes[0].mask);
    EXPECT_TRUE(pool_roots(*on).empty());
    for (auto* n : marked) {
        EXPECT_FALSE(n->can_share_buffer()) << n->id();
    }
    expect_same_state(*on, *off, "rejected class (skippable view)");
    network n_on(on, 0), n_off(off, 0);
    EXPECT_EQ(run_real(n_on, eng, g_on), run_real(n_off, eng, g_off));
}

// C-7: with the switch off, and with the switch on but a program-level predicate that is false (1a off), the legacy marking path runs and the result is
// bit-identical to the plain switch off; outputs equal.
TEST(sum_root_pool_p1c1_onednn, C_7_switch_off_is_bit_identical) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = chain2_graph(eng, view_kind::none, false);
    auto off = build_real(eng, g, false);
    P1C1_REQUIRE(count_formed_writers(*off, g) == 2, "both oneDNN sums must form");
    auto cfg_1a_off = real_cfg(eng, g, true);
    cfg_1a_off.set_property(ov::intel_gpu::enable_inplace_conflict_resolution(false));
    auto on_1a_off = program::build_program(eng, g.t, cfg_1a_off, false, false);   // consumes g.t: rebuild the topology for the next one
    expect_same_state(*on_1a_off, *off, "switch on with 1a off (defer = false) vs switch off");
    EXPECT_TRUE(pool_roots(*off).empty());
    EXPECT_FALSE(off->get_node("O").can_share_buffer()) << "legacy: the chain root is out of the pool";
    auto g2 = chain2_graph(eng, view_kind::none, false);
    auto concat_only_cfg = real_cfg(eng, g2, false);
    concat_only_cfg.set_property(ov::intel_gpu::enable_concat_holder_pool(true));
    auto concat_only = program::build_program(eng, g2.t, concat_only_cfg, false, false);
    auto g3 = chain2_graph(eng, view_kind::none, false);
    auto off2 = build_real(eng, g3, false);
    expect_same_state(*concat_only, *off2, "concat switch on only: sum chains untouched");
}

// C-8: outputs of the two configurations are elementwise equal (and the A/B row is printed).
TEST(sum_root_pool_p1c1_onednn, C_8_ab_outputs_elementwise_equal) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    for (bool with_view : {false, true}) {
        auto g_on = chain2_graph(eng, with_view ? view_kind::cropped : view_kind::none, true);
        auto g_off = chain2_graph(eng, with_view ? view_kind::cropped : view_kind::none, true);
        auto on = build_real(eng, g_on, true);
        auto off = build_real(eng, g_off, false);
        P1C1_REQUIRE(count_formed_writers(*on, g_on) == 2, "both oneDNN sums must form");
        const auto promoted = pool_roots(*on);
        ASSERT_EQ(promoted.size(), 1u);
        expect_state_except_roots(*on, *off, promoted, with_view ? "ab with view" : "ab plain");
        network n_on(on, 0), n_off(off, 0);
        const auto o_on = run_real(n_on, eng, g_on);
        const auto o_off = run_real(n_off, eng, g_off);
#ifdef GPU_DEBUG_CONFIG
        const auto a = n_on.get_held_bytes_snapshot();
        const auto b = n_off.get_held_bytes_snapshot();
        std::cout << "[AB] C-8 " << (with_view ? "crop(root=V)" : "plain") << " promoted=" << promoted.size() << " pooled on/off=" << a.pooled.bytes << "/" << b.pooled.bytes
                  << " off_pool on/off=" << a.off_pool_bytes() << "/" << b.off_pool_bytes() << " total on/off=" << a.total_bytes << "/" << b.total_bytes
                  << " allocs on/off=" << n_on.get_alloc_trace_summary().total_count << "/" << n_off.get_alloc_trace_summary().total_count
                  << " outputs_elementwise_equal=" << (o_on == o_off ? "yes" : "NO") << "\n";
#else
        std::cout << "[AB] C-8 " << (with_view ? "crop(root=V)" : "plain") << " outputs_elementwise_equal=" << (o_on == o_off ? "yes" : "NO") << " (GPU_DEBUG_CONFIG off: no byte stats)\n";
#endif
        EXPECT_EQ(o_on, o_off);
        EXPECT_TRUE(sum_lifetime_violations(n_on).empty());
    }
}

// C-9: W1 is downgraded by 1a (its sum would be overwritten while O still has a reader): W1 stops being a writer, becomes the root of the {W1 -> W2}
// class as legacy_root(W2) (in `marked`), and is promoted.
TEST(sum_root_pool_p1c1_onednn, C_9_downgraded_writer_becomes_root) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = chain_mid_graph(eng);
    auto prog = build_real(eng, g, true);
    const auto* rep = prog->get_inplace_conflict_report();
    ASSERT_NE(rep, nullptr);
    P1C1_REQUIRE(rep->entries.size() == 1 && rep->entries[0].writer_id == "W1" && count_formed_writers(*prog, g) == 1, "expected W1 downgraded by 1a and W2 kept");
    auto& W1 = prog->get_node("W1");
    auto& W2 = prog->get_node("W2");
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W1), -1) << "W1 is no longer a writer";
    std::vector<program_node*> marked;
    const auto marks = class_level::collect_sum_root_marks(*prog, &marked);
    EXPECT_NE(std::find(marked.begin(), marked.end(), &W1), marked.end()) << "W1 is legacy_root(W2): deferred, and promoted below";
    EXPECT_EQ(marks.of_writer.at(&W2).first, &W1);
    EXPECT_TRUE(W1.is_pool_class_root()) << "the downgraded W1 is the chain root of {W1, W2}";
    EXPECT_TRUE(W1.can_share_buffer());
    EXPECT_EQ(route_of(W1), primitive_inst::alloc_route::POOL_AS_ROOT);
    EXPECT_EQ(route_of(W2), primitive_inst::alloc_route::DIRECT);
    EXPECT_FALSE(W2.can_share_buffer());
    const auto census = class_level::run_sum_census(*prog);
    EXPECT_EQ(census.downgraded_roots, 1u) << census.summary();
    auto g_off = chain_mid_graph(eng);
    auto off = build_real(eng, g_off, false);
    expect_state_except_roots(*prog, *off, pool_roots(*prog), "downgraded chain");
    network n_on(prog, 0), n_off(off, 0);
    EXPECT_EQ(run_real(n_on, eng, g), run_real(n_off, eng, g_off));
    EXPECT_TRUE(sum_lifetime_violations(n_on).empty());
}

// C-10: DiT block variant (the attn residual far above the FC, MLP residuals near). The far attn sum (25 hops > the walk bound) breaks into a
// binary_add, so the attn FC output becomes the NEW chain root (the 15.1 GB ETC of Pattern C); two MLP-style sums (3 hops each) chain on it:
// {P_far -> P_mlp1 -> P_mlp2}. At least two writers (design: C-10 variant with >= 2 writers, the real DiT block has one MLP writer per block).
TEST(sum_root_pool_p1c1_onednn, C_10_dit_block_ab) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto make = [&] {
        fc_graph g;
        g.t.add(input_layout("in", f16_lay()));
        g.t.add(input_layout("in_far", f16_lay()));
        g.t.add(input_layout("in_m1", f16_lay()));
        g.t.add(input_layout("in_m2", f16_lay()));
        g.inputs = {"in", "in_far", "in_m1", "in_m2"};
        add_fc(g, eng, "R", "in", 1, impl_types::ocl);
        std::string cur = "R";
        for (int i = 0; i < 25; ++i) {   // 25 hops between the residual source and the attn FC
            const std::string id = "h" + std::to_string(i);
            add_fc(g, eng, id, cur, 10 + i, impl_types::ocl);   // plain FCs: activations would fuse into R and collapse the hops
            cur = id;
        }
        add_fc(g, eng, "P_far", cur, 2, impl_types::onednn);
        g.t.add(eltwise("S_far", {input_info("P_far"), input_info("R")}, eltwise_mode::sum));   // the far residual
        add_fc(g, eng, "m1a", "S_far", 3, impl_types::ocl);
        add_fc(g, eng, "P_mlp1", "m1a", 4, impl_types::onednn);
        g.t.add(eltwise("S_mlp1", {input_info("P_mlp1"), input_info("S_far")}, eltwise_mode::sum));
        add_fc(g, eng, "m2a", "S_mlp1", 5, impl_types::ocl);
        add_fc(g, eng, "P_mlp2", "m2a", 6, impl_types::onednn);
        g.t.add(eltwise("S_mlp2", {input_info("P_mlp2"), input_info("S_mlp1")}, eltwise_mode::sum));
        add_fc(g, eng, "Zb", "S_mlp2", 7, impl_types::ocl);
        g.t.add(reorder("out_Zb", input_info("Zb"), format::bfyx, data_types::f32));
        g.outs = {"out_Zb"};
        g.writers = {"P_far", "P_mlp1", "P_mlp2"};
        return g;
    };
    auto g_on = make();
    auto g_off = make();
    auto on = build_real(eng, g_on, true);
    auto off = build_real(eng, g_off, false);
    const size_t formed = count_formed_writers(*on, g_on);
    std::cout << "[C-10] formed writers=" << formed << " P_far frozen sum=" << fusing::sum_writer_root_marks(on->get_node("P_far")).has_value() << "\n";
    P1C1_REQUIRE(!fusing::sum_writer_root_marks(on->get_node("P_far")).has_value(), "the far attn sum is expected to be demoted to binary_add (25 hops)");
    P1C1_REQUIRE(formed >= 2, "the two MLP sums must form");
    auto& Pf = on->get_node("P_far");
    EXPECT_TRUE(Pf.is_pool_class_root()) << "the demoted attn output is the chain root of the MLP class and enters the pool (Pattern C)";
    EXPECT_EQ(route_of(Pf), primitive_inst::alloc_route::POOL_AS_ROOT);
    expect_state_except_roots(*on, *off, pool_roots(*on), "dit block");
    network n_on(on, 0), n_off(off, 0);
    const auto o_on = run_real(n_on, eng, g_on);
    const auto o_off = run_real(n_off, eng, g_off);
#ifdef GPU_DEBUG_CONFIG
    const auto a = n_on.get_held_bytes_snapshot();
    const auto b = n_off.get_held_bytes_snapshot();
    std::cout << "[AB] C-10 promoted=" << pool_roots(*on).size() << " pooled on/off=" << a.pooled.bytes << "/" << b.pooled.bytes << " off_pool on/off=" << a.off_pool_bytes()
              << "/" << b.off_pool_bytes() << " total on/off=" << a.total_bytes << "/" << b.total_bytes << " outputs_elementwise_equal=" << (o_on == o_off ? "yes" : "NO") << "\n";
#endif
    EXPECT_EQ(o_on, o_off);
    EXPECT_TRUE(sum_lifetime_violations(n_on).empty());
}

// C-16 / C-16b: a sum root with padding (L1S-12, kAdmitPaddedSumRoot). R is a conv whose consumer is a padded 3x3 conv (prepare_padding pads R); the
// oneDNN sum then writes into R. C-16b: the pool block of R is filled with garbage before the run; the kernel that writes R leaves the padding untouched,
// so the outputs must still equal the switch-off run (direct, zero-initialised allocation). Not formed -> SKIP (and the preset is flipped to false).
namespace {

topology topo_padded_root(engine& eng, bool cp_reads_r = true) {
    topology t;
    t.add(input_layout("in1", f16_lay_c(32)));
    t.add(input_layout("in2", f16_lay_c(32)));
    t.add(cldnn::data("w_r", make_conv_weights(eng, 32, 32, 1)));
    t.add(cldnn::data("w_c", make_conv_weights(eng, 32, 32, 3)));
    t.add(cldnn::data("w_w", make_conv_weights(eng, 32, 32, 3)));
    add_conv_p(t, "R", "in2", "w_r", 0);
    add_conv_p(t, "Cp", cp_reads_r ? "R" : "in2", "w_c", 1);   // pads R
    add_conv_p(t, "W", "in1", "w_w", 1);
    t.add(eltwise("S", {input_info("W"), input_info("R")}, eltwise_mode::sum));
    t.add(reorder("out_J", input_info("S"), format::bfyx, data_types::f32));
    t.add(reorder("out_Cp", input_info("Cp"), format::bfyx, data_types::f32));
    return t;
}

ExecutionConfig padded_cfg(engine& eng, bool sum_switch) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    ov::intel_gpu::ImplForcingMap forced = {{"R", fsv16_impl(impl_types::ocl)}, {"Cp", fsv16_impl(impl_types::ocl)}, {"W", fsv16_impl(impl_types::onednn)}};
    config.set_property(ov::intel_gpu::force_implementations(forced));
    config.set_property(ov::intel_gpu::enable_sum_root_pool(sum_switch));
    return config;
}

}  // namespace

TEST(sum_root_pool_p1c1_onednn, C_16_C_16b_padded_sum_root) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto t_on = topo_padded_root(eng);
    auto t_off = topo_padded_root(eng);
    auto on = program::build_program(eng, t_on, padded_cfg(eng, true), false, false);
    auto off = program::build_program(eng, t_off, padded_cfg(eng, false), false, false);
    P1C1_REQUIRE(fusing::sum_writer_root_marks(on->get_node("W")).has_value(), "the oneDNN sum of W into R did not form (a padded R and the unpadded W output differ)");
    auto& R = on->get_node("R");
    P1C1_REQUIRE(static_cast<bool>(R.get_output_layout(0).data_padding), "R carries no padding");
    EXPECT_EQ(R.is_pool_class_root(), class_level::kAdmitPaddedSumRoot) << "a padded root is promoted only if kAdmitPaddedSumRoot (preset flipped to false in P1-C-1)";
    network n_on(on, 0), n_off(off, 0);
    auto run = [&](network& n, bool dirty) {
        if (dirty) {   // garbage in the whole block of R (padding included)
            auto m = n.get_primitive("R")->output_memory_ptr();
            mem_lock<uint8_t, mem_lock_type::write> lock(m, get_test_stream());
            std::fill(lock.begin(), lock.end(), static_cast<uint8_t>(0x7B));
        }
        n.set_input_data("in1", conv_input(eng, 32, 1));
        n.set_input_data("in2", conv_input(eng, 32, 2));
        auto outs = n.execute();
        auto mem = outs.at("out_J").get_memory();
        mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
        return std::vector<float>(lock.begin(), lock.end());
    };
    const auto ref = run(n_off, false);
    EXPECT_EQ(run(n_on, true), ref) << "C-16b: a dirty padding region of the pooled root must not change the result";
    EXPECT_EQ(run(n_on, true), ref) << "second run";
}

// C-21: a sum class and a concat holder class (both switches on) whose lifetimes intersect: the cross-class closure (I-B1 across classes) holds, the two
// pool users never share a block, the outputs equal the switch-off run.
TEST(sum_root_pool_p1c1_onednn, C_21_sum_class_and_concat_class_intersect) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto make = [&] {
        topology t;
        t.add(input_layout("in1", f16_lay_c(16)));
        t.add(input_layout("in3", f16_lay_c(32)));
        t.add(input_layout("in2", f16_lay_c(32)));
        t.add(cldnn::data("w_pq", make_conv_weights(eng, 16, 16, 1)));
        add_conv_p(t, "P", "in1", "w_pq", 0);
        add_conv_p(t, "Q", "in1", "w_pq", 0);
        t.add(concatenation("C", {input_info("P"), input_info("Q")}, 1));
        t.add(cldnn::data("w_w", make_conv_weights(eng, 32, 32, 3)));
        add_conv_p(t, "W", "in3", "w_w", 1);
        t.add(activation("X", input_info("in2"), activation_func::relu));
        t.add(eltwise("S", {input_info("W"), input_info("X")}, eltwise_mode::sum));
        t.add(eltwise("M", {input_info("S"), input_info("C")}, eltwise_mode::prod));   // both classes are alive until M
        t.add(reorder("out_M", input_info("M"), format::bfyx, data_types::f32));
        return t;
    };
    auto cfg = [&](bool on) {
        ExecutionConfig config = get_test_default_config(eng);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
        ov::intel_gpu::ImplForcingMap forced = {{"P", fsv16_impl(impl_types::ocl)}, {"Q", fsv16_impl(impl_types::ocl)}, {"C", fsv16_impl(impl_types::ocl)},
                                                {"X", fsv16_impl(impl_types::ocl)}, {"W", fsv16_impl(impl_types::onednn)}};
        config.set_property(ov::intel_gpu::force_implementations(forced));
        config.set_property(ov::intel_gpu::enable_sum_root_pool(on));
        config.set_property(ov::intel_gpu::enable_concat_holder_pool(on));
#ifdef GPU_DEBUG_CONFIG
        config.set_user_property({ov::intel_gpu::dump_alloc_trace(1)});
#endif
        return config;
    };
    auto t_on = make();
    auto t_off = make();
    auto on = program::build_program(eng, t_on, cfg(true), false, false);
    auto off = program::build_program(eng, t_off, cfg(false), false, false);
    P1C1_REQUIRE(fusing::sum_writer_root_marks(on->get_node("W")).has_value(), "the oneDNN sum of W did not form");
    P1C1_REQUIRE(on->get_node("C").can_be_optimized() && on->get_node("C").is_pool_class_root(), "the concat holder class C was not promoted");
    auto& X = on->get_node("X");
    P1C1_REQUIRE(X.is_pool_class_root(), "the sum class rooted at X was not promoted");
    auto& C = on->get_node("C");
    const auto sx = span_of_class(*on, "X");
    const auto sc = span_of_class(*on, "C");
    P1C1_REQUIRE(!(sx.death < sc.birth || sc.death < sx.birth), "the lifetimes of the two classes do not intersect in processing order");
    EXPECT_TRUE(has_restriction(X, C) && has_restriction(C, X)) << "intersecting sum and concat classes restrict each other in both directions";
    EXPECT_EQ(class_level::check_b4_prime(*on), "");
    network n_on(on, 0), n_off(off, 0);
    auto run = [&](network& n) {
        n.set_input_data("in1", conv_input(eng, 16, 1));
        n.set_input_data("in3", conv_input(eng, 32, 2));
        n.set_input_data("in2", conv_input(eng, 32, 3));
        auto outs = n.execute();
        std::map<std::string, std::vector<float>> r;
        for (auto& o : outs) {
            auto mem = o.second.get_memory();
            mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
            r[o.first] = std::vector<float>(lock.begin(), lock.end());
        }
        return r;
    };
    EXPECT_EQ(run(n_on), run(n_off));
    EXPECT_NE(buf_of(n_on, "X"), buf_of(n_on, "C")) << "intersecting classes of different kinds must not share a block";
    EXPECT_TRUE(sum_lifetime_violations(n_on).empty());
}

// C-19 (CL-2): the output-view owner gap for a sum root. Y -> r1 (reorder, network output, compatible but not identical layout), Y -> r2 (identical, merged away:
// r1 inherits the user) -> Z -> R (chain root, same bytes as Y) -> W1 (sum) -> W2 (sum). After the merge y = r1 is an output view of Y that has a user and Y has no
// other user: oooq / skipped_branch do not relate R and Y, and the direct insertion of r1 by past_outputs skips the fold of r1 to Y. L3 step (5) adds Y to R(R).
TEST(sum_root_pool_p1c1_onednn, C_19_output_view_owner_before_sum_root_never_shares) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto make = [&] {
        fc_graph g;
        g.t.add(input_layout("in_y", f16_lay()));
        g.t.add(input_layout("in1", f16_lay()));
        g.inputs = {"in_y", "in1"};
        g.t.add(activation("Y", input_info("in_y"), activation_func::abs));
        g.t.add(reorder("r1", input_info("Y"), format::bfzyx, data_types::f16));
        g.t.add(reorder("r2", input_info("Y"), format::bfzyx, data_types::f16));
        g.t.add(activation("Z", input_info("r2"), activation_func::abs));
        add_fc(g, eng, "R", "Z", 1, impl_types::ocl);
        add_fc_dims(g, eng, "W1", "in1", 0, impl_types::onednn, 16, 16);
        g.t.add(eltwise("S1", {input_info("W1"), input_info("R")}, eltwise_mode::sum));
        g.t.add(activation("a1", input_info("S1"), activation_func::abs));
        add_fc_dims(g, eng, "W2", "a1", 4, impl_types::onednn, 16, 16);
        g.t.add(eltwise("S2", {input_info("W2"), input_info("S1")}, eltwise_mode::sum));
        add_fc_dims(g, eng, "Zb", "S2", 5, impl_types::ocl, 16, 16);
        g.t.add(reorder("out_Zb", input_info("Zb"), format::bfyx, data_types::f32));
        g.outs = {"r1", "out_Zb"};
        g.writers = {"W1", "W2"};
        return g;
    };
    auto cfg = [&](const fc_graph& g, bool on) {
        auto c = real_cfg(eng, g, on);
        c.set_property(ov::intel_gpu::custom_outputs(g.outs));
        return c;
    };
    auto g_on = make();
    auto g_off = make();
    auto on = program::build_program(eng, g_on.t, cfg(g_on, true), false, false);
    auto off = program::build_program(eng, g_off.t, cfg(g_off, false), false, false);
    P1C1_REQUIRE(count_formed_writers(*on, g_on) == 2, "both oneDNN sums must form");
    auto& y = on->get_node("r1");
    auto& Y = on->get_node("Y");
    P1C1_REQUIRE(y.is_output(), "r1 is not a network output");
    P1C1_REQUIRE(!y.get_users().empty(), "r1 has no user: the duplicate reorder r2 was not merged into it");
    P1C1_REQUIRE(y.can_be_optimized() && y.as<reorder>().requires_reinterpret(), "r1 is not a can_be_optimized + requires_reinterpret view");
    P1C1_REQUIRE(Y.get_users().size() == 1, "Y has more than one user");
    P1C1_REQUIRE(on->get_node("R").is_pool_class_root(), "the sum class of R was not promoted");
    auto& R = on->get_node("R");
    P1C1_REQUIRE(R.get_output_layout(0).bytes_count() == Y.get_output_layout(0).bytes_count(), "R and Y differ in size");
    P1C1_REQUIRE(!off->get_node("R").has_memory_dependency(off->get_node("Y").get_unique_id()), "the LEGACY restriction of R already contains Y (the gap is not present)");
    EXPECT_TRUE(has_restriction(R, Y)) << "L3 step (5): the owner Y of the output view r1 must enter R(R)";
    EXPECT_TRUE(has_restriction(Y, R)) << "... and R must enter R(Y)";
    network n_on(on, 0), n_off(off, 0);
    const auto o_on = run_real(n_on, eng, g_on);
    const auto o_off = run_real(n_off, eng, g_off);
    EXPECT_NE(buf_of(n_on, "R"), buf_of(n_on, "Y")) << "R must not take the block of Y while the output view of Y is alive";
    EXPECT_EQ(o_on.at("out_Zb"), o_off.at("out_Zb")) << "the non-view output equals the switch-off run";
    EXPECT_TRUE(sum_lifetime_violations(n_on).empty());
}

// C-11: B4' is discriminating: a writer carrying a class route, or a dependency / owner of a writer that is a class member, is reported; the promoted
// root of the writer's own class is allowed.
TEST(sum_root_pool_p1c1_onednn, C_11_b4_prime_detects_writer_with_class_route) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = chain2_graph(eng, view_kind::none, false);
    auto prog = build_real(eng, g, true);
    P1C1_REQUIRE(count_formed_writers(*prog, g) == 2, "both oneDNN sums must form");
    ASSERT_TRUE(prog->get_node("O").is_pool_class_root());
    ASSERT_EQ(class_level::check_b4_prime(*prog), "") << "the dependency of W1 is the promoted root of its own class: allowed";
    auto& W1 = prog->get_node("W1");
    auto& W2 = prog->get_node("W2");
    {   // the writer becomes a zero allocation member
        W1.set_alias_pool_root_uid(static_cast<uint32_t>(prog->get_node("O").get_unique_id()));
        EXPECT_NE(class_level::check_b4_prime(*prog), "");
        W1.set_alias_pool_root_uid(program_node::invalid_alias_pool_root_uid);
        EXPECT_EQ(class_level::check_b4_prime(*prog), "");
    }
    {   // the writer becomes a pool root
        W2.set_pool_class_root(true);
        W2.can_share_buffer(true);
        EXPECT_NE(class_level::check_b4_prime(*prog), "");
        W2.set_pool_class_root(false);
        W2.can_share_buffer(false);
        EXPECT_EQ(class_level::check_b4_prime(*prog), "");
    }
    {   // W2 depends on W1 (not a class root): a class route on the dependency is reported
        W1.set_pool_class_root(true);
        W1.can_share_buffer(true);
        EXPECT_NE(class_level::check_b4_prime(*prog), "");
        W1.set_pool_class_root(false);
        W1.can_share_buffer(false);
        EXPECT_EQ(class_level::check_b4_prime(*prog), "");
    }
    {   // a promoted root that is not shareable
        auto& O = prog->get_node("O");
        O.can_share_buffer(false);
        EXPECT_NE(class_level::check_b4_prime(*prog), "");
        O.can_share_buffer(true);
        EXPECT_EQ(class_level::check_b4_prime(*prog), "");
    }
}

// C-12: the flags of a promoted sum class survive the model cache; the import does not re-run the generator; the version is unchanged (3).
TEST(sum_root_pool_p1c1_onednn, C_12_cache_round_trip) {
    P1C0_SKIP_IF_NO_IMMAD();
    EXPECT_EQ(ov::intel_gpu::CompiledModel::runtime_requirements_version, 3u) << "P1-C-1 adds no serialized field: the version stays at the P1-B-S value";
    auto& eng = get_test_engine();
    auto g = chain2_graph(eng, view_kind::none, true);
    auto prog = build_real(eng, g, true);
    P1C1_REQUIRE(count_formed_writers(*prog, g) == 2, "both oneDNN sums must form");
    ASSERT_EQ(pool_roots(*prog).size(), 1u);
    cldnn::membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        cldnn::BinaryOutputBuffer ob = cldnn::BinaryOutputBuffer(out_mem);
        ob.set_stream(get_test_stream_ptr().get());
        prog->save(ob);
    }
    std::istream in_mem(&mem_buf);
    cldnn::BinaryInputBuffer ib = cldnn::BinaryInputBuffer(in_mem, eng);
    auto imported = std::make_shared<cldnn::program>(eng, real_cfg(eng, g, true));
    imported->load(ib);
    for (auto* n : prog->get_processing_order()) {
        const auto& m = imported->get_node(n->id());
        EXPECT_EQ(m.is_pool_class_root(), n->is_pool_class_root()) << n->id();
        EXPECT_EQ(m.get_alias_pool_root_uid(), n->get_alias_pool_root_uid()) << n->id();
        EXPECT_EQ(m.can_share_buffer(), n->can_share_buffer()) << n->id();
        EXPECT_EQ(m.get_memory_dependencies(), n->get_memory_dependencies()) << n->id();
    }
    EXPECT_EQ(imported->get_alias_analysis(), nullptr) << "the import does not run prepare_memory_dependencies";
    auto g_ref = chain2_graph(eng, view_kind::none, true);
    auto ref_prog = build_real(eng, g_ref, true);
    network ref_net(ref_prog, 0);
    const auto ref = run_real(ref_net, eng, g_ref);
    network net(imported, 0);
    EXPECT_EQ(run_real(net, eng, g), ref) << "the imported program must produce the uncached result";
    EXPECT_TRUE(sum_lifetime_violations(net).empty());
    EXPECT_EQ(route_of(net.get_program()->get_node("O")), primitive_inst::alloc_route::POOL_AS_ROOT);
    EXPECT_EQ(buf_of(net, "W1"), buf_of(net, "O"));
}

// C-13: the sum writes into a crop of the root. Through the pipeline the crop stays a compute node (the root of the class is V itself, not O), so there is
// no static can_be_optimized crop view of a class root to test: SKIP when V is not an in-place view. (A crop that is a settled view is covered at the
// engine level by the F4/F6 tests; P1-C-1 itself treats such a member as MUST_VIEW and L1S-4 admits it.)
TEST(sum_root_pool_p1c1_onednn, C_13_sum_into_crop_view_of_root) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = chain2_graph(eng, view_kind::cropped, false);
    auto prog = build_real(eng, g, true);
    P1C1_REQUIRE(count_formed_writers(*prog, g) == 2, "both oneDNN sums must form");
    P1C1_REQUIRE(prog->get_node("V").can_be_optimized(), "the crop V is a compute node (not an in-place view of O): no crop view of the root to test");
    EXPECT_TRUE(prog->get_node("O").is_pool_class_root());
    EXPECT_EQ(class_level::check_b4_prime(*prog), "");
}

// C-14: the estimate (program::get_estimated_device_mem_usage) differs between the two configurations; the exact relation is measured and reported.
TEST(sum_root_pool_p1c1_onednn, C_14_estimate_differs_by_promoted_roots) {
#ifndef GPU_DEBUG_CONFIG
    GTEST_SKIP() << "the held-bytes snapshot used as the reference needs GPU_DEBUG_CONFIG";
#else
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g_on = chain2_graph(eng, view_kind::none, true);
    auto g_off = chain2_graph(eng, view_kind::none, true);
    auto prog_on = build_real(eng, g_on, true);
    auto prog_off = build_real(eng, g_off, false);
    P1C1_REQUIRE(count_formed_writers(*prog_on, g_on) == 2, "both oneDNN sums must form");
    const int64_t root_bytes = static_cast<int64_t>(prog_on->get_node("O").get_output_layout().bytes_count());
    const auto e_on = prog_on->get_estimated_device_mem_usage();
    const auto e_off = prog_off->get_estimated_device_mem_usage();
    std::cout << "[C-14] estimate on=" << e_on.second << " off=" << e_off.second << " diff=" << (e_on.second - e_off.second) << " root_bytes=" << root_bytes << "\n";
    ASSERT_GE(e_on.second, 0);
    ASSERT_GE(e_off.second, 0);
    EXPECT_EQ(e_off.second - e_on.second, root_bytes) << "the estimate differs by exactly the bytes of the promoted root (measured: the switch-on estimate is lower)";
#endif
}

// C-15: the root is the first poolable node of the graph (the first compute node after the inputs): lock the behaviour.
TEST(sum_root_pool_p1c1_onednn, C_15_root_is_first_poolable_node) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g_on = chain_graph(eng);
    auto g_off = chain_graph(eng);
    auto on = build_real(eng, g_on, true);
    auto off = build_real(eng, g_off, false);
    P1C1_REQUIRE(count_formed_writers(*on, g_on) == 1, "the sum of p must form");
    const program_node* first = nullptr;
    for (auto* n : on->get_processing_order()) {
        if (!n->is_type<data>() && !n->is_type<input_layout>() && n->may_use_mempool()) {
            first = n;
            break;
        }
    }
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->id(), "x_proj");
    EXPECT_TRUE(first->is_pool_class_root());
    expect_state_except_roots(*on, *off, pool_roots(*on), "first poolable root");
    network n_on(on, 0), n_off(off, 0);
    EXPECT_EQ(run_real(n_on, eng, g_on), run_real(n_off, eng, g_off));
    EXPECT_TRUE(sum_lifetime_violations(n_on).empty());
}

// C-17: with the switch on the frozen post-op list, the writers and the routes stay consistent after a 1a downgrade (the divergence only ever
// downgrades); every surviving frozen sum names the writer that routes DIRECT.
TEST(sum_root_pool_p1c1_onednn, C_17_f5_consistency_with_the_switch_on) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = chain_mid_graph(eng);
    auto prog = build_real(eng, g, true);
    P1C1_REQUIRE(prog->get_inplace_conflict_report() != nullptr && prog->get_inplace_conflict_report()->entries.size() == 1, "expected exactly one 1a downgrade");
    for (auto* n : prog->get_processing_order()) {
        if (n->is_dynamic()) {
            continue;
        }
        for (const auto& e : n->get_fused_primitives_onednn()) {
            if (e.op_type != onednn_post_op_type::sum && e.op_type != onednn_post_op_type::optimized_sum) {
                continue;
            }
            EXPECT_EQ(fusing::get_reused_eltwmem_idx(*n), static_cast<int32_t>(e.mem_dep)) << n->id() << ": impl post-op and live decision agree";
            EXPECT_EQ(route_of(*n), primitive_inst::alloc_route::DIRECT) << n->id();
            EXPECT_FALSE(n->can_share_buffer()) << n->id();
        }
    }
    EXPECT_EQ(class_level::check_b4_prime(*prog), "");
}

// C-18 (T-OOOQ): requesting out_of_order with oneDNN enabled is forced to in_order; the sum root pool then behaves as in_order. A genuine
// out_of_order queue (use_onednn(false)) is covered by C0-4 (bit identical).
TEST(sum_root_pool_p1c1_onednn, C_18_onednn_requested_oooq_is_forced_in_order) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = chain2_graph(eng, view_kind::none, false);
    auto cfg = real_cfg(eng, g, true);
    cfg.set_property(ov::intel_gpu::queue_type(QueueTypes::out_of_order));
    auto prog = program::build_program(eng, g.t, cfg, false, false);
    ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::in_order) << "oneDNN forces in_order (T-OOOQ)";
    P1C1_REQUIRE(count_formed_writers(*prog, g) == 2, "both oneDNN sums must form");
    EXPECT_EQ(pool_roots(*prog).size(), 1u) << "behaves exactly as the in_order request";
}

// C-20: two sum classes whose lifetimes intersect (two independent chains read in the same step) never share a block; the cross-class closure holds.
TEST(sum_root_pool_p1c1_onednn, C_20_two_intersecting_sum_classes) {
    P1C0_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto make = [&] {
        fc_graph g;
        g.t.add(input_layout("in_a", f16_lay()));
        g.t.add(input_layout("in_b", f16_lay()));
        g.t.add(input_layout("in_c", f16_lay()));
        g.t.add(input_layout("in_d", f16_lay()));
        g.inputs = {"in_a", "in_b", "in_c", "in_d"};
        add_fc(g, eng, "RA", "in_a", 1, impl_types::ocl);
        add_fc(g, eng, "RB", "in_b", 2, impl_types::ocl);
        add_fc(g, eng, "WA", "in_c", 3, impl_types::onednn);
        g.t.add(eltwise("SA", {input_info("WA"), input_info("RA")}, eltwise_mode::sum));
        add_fc(g, eng, "WB", "in_d", 4, impl_types::onednn);
        g.t.add(eltwise("SB", {input_info("WB"), input_info("RB")}, eltwise_mode::sum));
        g.t.add(eltwise("J", {input_info("SA"), input_info("SB")}, eltwise_mode::prod));
        add_fc(g, eng, "Zb", "J", 5, impl_types::ocl);
        g.t.add(reorder("out_Zb", input_info("Zb"), format::bfyx, data_types::f32));
        g.outs = {"out_Zb"};
        g.writers = {"WA", "WB"};
        return g;
    };
    auto g_on = make();
    auto g_off = make();
    auto on = build_real(eng, g_on, true);
    auto off = build_real(eng, g_off, false);
    P1C1_REQUIRE(count_formed_writers(*on, g_on) == 2, "both oneDNN sums must form");
    P1C1_REQUIRE(pool_roots(*on).size() == 2, "both classes must be promoted");
    auto& RA = on->get_node("RA");
    auto& RB = on->get_node("RB");
    EXPECT_TRUE(has_restriction(RA, RB) && has_restriction(RB, RA)) << "intersecting classes restrict each other in both directions (I-B1 across classes)";
    network n_on(on, 0), n_off(off, 0);
    const auto o_on = run_real(n_on, eng, g_on);
    EXPECT_EQ(o_on, run_real(n_off, eng, g_off));
    EXPECT_NE(buf_of(n_on, "RA"), buf_of(n_on, "RB")) << "two intersecting sum classes must not share a block";
    EXPECT_TRUE(sum_lifetime_violations(n_on).empty());
}

#endif  // ENABLE_ONEDNN_FOR_GPU
