// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Phase 0.5a: alias_analysis engine core and the defect B transition marking in basic_memory_dependencies.
// Design: /home/ceciliapeng/repos/reports/memman/layer1/11_p1_design_r2.zh-CN.md section 2 (r3) and 08 B.8 (A1-A16).
//
// Two kinds of graphs are used:
//   * "hand" graphs: built with no_optimizations, the can_be_optimized / runtime_skippable / output / preferred impl flags are set by the
//     test itself. They exercise the engine as a pure function of node flags, without depending on any optimization pass;
//   * "pipeline" graphs: built by the regular program::build_program(), the flags come from the real passes. Used for the wiring in
//     prepare_memory_dependencies, the recursion-consistency check (A13) and the neutrality check (A14).
// Tests that need a oneDNN in-place writer are GTEST_SKIPped without supports_immad (a SKIP counts as "not covered").

#include "test_utils.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "alias_analysis.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/arg_max_min.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/kv_cache.hpp"
#include "intel_gpu/primitives/lora.hpp"
#include "intel_gpu/primitives/stateless_kv.hpp"
#include "primitive_type.h"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/non_max_suppression.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/primitives/softmax.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "pass_manager.h"
#include "program_helpers.h"
#include "program_node.h"
#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

namespace {

// ------------------------------------------------------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------------------------------------------------------
ExecutionConfig make_cfg(engine& eng, QueueTypes q = QueueTypes::in_order, bool optimize = false, const std::vector<std::string>& outputs = {}) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(optimize));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(q));
    if (!outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(outputs));
    }
    if (q == QueueTypes::out_of_order) {
        // With oneDNN enabled (Arc), ExecutionConfig::finalize() forces the in-order queue, so oneDNN must be switched off. It has to be a
        // *user* property: set_property() with the default value is not recorded as user-set and finalize() would re-enable oneDNN.
        config.set_user_property({ov::intel_gpu::use_onednn(false)}, ov::OptionVisibility::RELEASE_INTERNAL);
    }
    return config;
}

// A "hand" graph: no optimization passes, every node has valid output layouts, flags are set by the test.
program::ptr make_hand(engine& eng, const topology& t, const std::vector<std::string>& outs = {}, QueueTypes q = QueueTypes::in_order,
                       bool internal = false, bool body = false) {
    auto prog = program::build_program(eng, t, make_cfg(eng, q, false, outs), internal, /*no_optimizations=*/true, body);
    for (auto* n : prog->get_processing_order()) {
        n->get_output_layouts(false);
        n->set_unique_id();  // compile_graph assigns the ids; without that stage all nodes would share id 0
    }
    return prog;
}

const alias_member* find_member(const alias_class& c, const program_node& n, uint16_t port = 0) {
    for (const auto& m : c.members) {
        if (m.node == &n && m.id.index == port) {
            return &m;
        }
    }
    return nullptr;
}

std::string reader_ids(const alias_class& c) {
    std::ostringstream os;
    for (const auto* r : c.readers) {
        os << r->id() << " ";
    }
    return os.str();
}

std::string dump_aa(const alias_analysis& aa) {
    std::ostringstream os;
    for (const auto& c : aa.classes()) {
        os << "class root=" << c.root_node->id() << ":" << c.root.index << " fb=0x" << std::hex << c.fallback << " dm=0x" << c.domain << std::dec
           << " place=" << static_cast<int>(c.get_placement()) << " members=[";
        for (const auto& m : c.members) {
            os << m.node->id() << ":" << m.id.index << "/k" << static_cast<int>(m.kind) << "/o" << static_cast<int>(m.off) << "/r"
               << static_cast<int>(m.role) << " ";
        }
        os << "] readers=[" << reader_ids(c) << "]\n";
    }
    return os.str();
}

std::string dump_flags(program& prog) {
    std::ostringstream os;
    for (auto* n : prog.get_processing_order()) {
        os << "[FLAGS] #" << prog.get_processing_order().get_processing_number(n) << " " << n->id() << " type=" << n->get_primitive()->type_string()
           << " cbo=" << n->can_be_optimized() << " rs=" << n->is_runtime_skippable() << " out=" << n->is_output()
           << " cbs=" << n->can_share_buffer() << " users=" << n->get_users().size() << "\n";
    }
    return os.str();
}

#define GET_CLASS(var, aa, prog, nid)                                      \
    const alias_class* var = (aa).class_of((prog)->get_node(nid));         \
    ASSERT_NE(var, nullptr) << "no class for " << nid << "\n" << dump_aa(aa)

#define EXPECT_KIND(cls, prog, nid, expected_kind, expected_off, expected_role)                                             \
    do {                                                                                                                   \
        const auto* m__ = find_member(*(cls), (prog)->get_node(nid));                                                      \
        ASSERT_NE(m__, nullptr) << nid << " is not a member of the class of " << (cls)->root_node->id();                    \
        EXPECT_EQ(m__->kind, expected_kind) << nid;                                                                        \
        EXPECT_EQ(m__->off, expected_off) << nid;                                                                          \
        EXPECT_EQ(m__->role, expected_role) << nid;                                                                         \
    } while (0)

layout f32_l(const ov::PartialShape& s = ov::PartialShape{1, 4, 8, 8}) {
    return layout{s, data_types::f32, format::bfyx};
}

// in -> P -> R(view) -> Q(view) -> O (output)
topology chain_topology() {
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(activation("R", input_info("P"), activation_func::abs));
    t.add(activation("Q", input_info("R"), activation_func::abs));
    t.add(activation("O", input_info("Q"), activation_func::abs));
    return t;
}

// in1 -> P1 --+
//             C (axis 1) -> O
// in2 -> P2 --+
topology concat_topology(const layout& in_l = f32_l(), data_types dt = data_types::f32) {
    topology t;
    t.add(input_layout("in1", in_l));
    t.add(input_layout("in2", in_l));
    t.add(activation("P1", input_info("in1"), activation_func::relu));
    t.add(activation("P2", input_info("in2"), activation_func::abs));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1, dt));
    t.add(activation("O", input_info("C"), activation_func::abs));
    return t;
}

// Pipeline concat graph (real prepare_buffer_fusing makes C an in-place concat): reorder -> concat -> reorder.
topology pipeline_concat_topology() {
    topology t;
    t.add(input_layout("in1", layout{ov::PartialShape{1, 16, 8, 8}, data_types::f32, format::bfyx}));
    t.add(input_layout("in2", layout{ov::PartialShape{1, 16, 8, 8}, data_types::f32, format::bfyx}));
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(reorder("out", input_info("C"), format::bfyx, data_types::f32));
    // an independent sibling branch
    t.add(input_layout("in3", layout{ov::PartialShape{1, 16, 8, 8}, data_types::f32, format::bfyx}));
    t.add(activation("B1", input_info("in3"), activation_func::relu));
    t.add(activation("B2", input_info("B1"), activation_func::abs));
    t.add(reorder("out2", input_info("B2"), format::bfyx, data_types::f32));
    return t;
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

// Node-identity lookup must be exact for every node even when uids collide; the uid-keyed lookup is exact for unique uids and nullptr for
// colliding ones.
void expect_identity_lookup_exact(program& prog, const alias_analysis& aa) {
    std::map<uint32_t, size_t> multiplicity;
    for (auto* n : prog.get_processing_order()) {
        ++multiplicity[static_cast<uint32_t>(n->get_unique_id())];
    }
    for (auto* n : prog.get_processing_order()) {
        const alias_class* c = aa.class_of(*n);
        ASSERT_NE(c, nullptr) << n->id();
        ASSERT_NE(find_member(*c, *n, 0), nullptr) << "class_of(node) does not contain the node " << n->id() << "\n" << dump_aa(aa);
        const uint32_t uid = static_cast<uint32_t>(n->get_unique_id());
        const alias_class* by_uid = aa.class_of(tensor_id{uid, tensor_space::OUTPUT, 0});
        if (multiplicity[uid] == 1) {
            EXPECT_EQ(by_uid, c) << n->id();
        } else {
            EXPECT_EQ(by_uid, nullptr) << "colliding uid of " << n->id() << " must be ambiguous";
        }
    }
}

std::string uid_to_id(program& prog, uint32_t uid) {
    for (auto* n : prog.get_processing_order()) {
        if (static_cast<uint32_t>(n->get_unique_id()) == uid) {
            return n->id();
        }
    }
    return "?" + std::to_string(uid);
}

// restriction sets (by node id) and can_share_buffer flags
struct snapshot {
    std::map<std::string, std::set<std::string>> restrictions;
    std::map<std::string, bool> shareable;
    bool operator==(const snapshot& o) const {
        return restrictions == o.restrictions && shareable == o.shareable;
    }
};

snapshot take_snapshot(program& prog) {
    snapshot s;
    for (auto* n : prog.get_processing_order()) {
        auto& r = s.restrictions[n->id()];
        for (auto uid : n->get_memory_dependencies()) {
            r.insert(uid_to_id(prog, uid));
        }
        s.shareable[n->id()] = n->can_share_buffer();
    }
    return s;
}

std::string diff_snapshot(const snapshot& a, const snapshot& b) {
    std::ostringstream os;
    for (const auto& kv : a.restrictions) {
        auto it = b.restrictions.find(kv.first);
        if (it == b.restrictions.end() || it->second != kv.second) {
            os << "restriction differs for " << kv.first << "\n";
        }
    }
    for (const auto& kv : a.shareable) {
        auto it = b.shareable.find(kv.first);
        if (it == b.shareable.end() || it->second != kv.second) {
            os << "can_share_buffer differs for " << kv.first << " (" << kv.second << " vs " << (it == b.shareable.end() ? -1 : int(it->second)) << ")\n";
        }
    }
    return os.str();
}

enum class pmd_mode { none, legacy_only, full };

// Same stages as program::build_program(), then prepare_memory_dependencies either completely (with the alias_analysis build), or as
// the three legacy passes only. NOTE: legacy_only runs the CURRENT basic_memory_dependencies (including the defect B owner mark), so it is
// NOT the HEAD behaviour. It only isolates the alias_analysis::build call (pure read) from the rest of prepare_memory_dependencies.
// pmd_mode::none stops before any dependency pass and is the base for the HEAD re-implementation below.
program::ptr build_to_pmd(engine& eng, const topology& t, const ExecutionConfig& cfg, pmd_mode mode) {
    auto prog = program::build_program(eng, t, cfg, /*is_internal=*/false, /*no_optimizations=*/true);
    program_wrapper::pre_optimize_graph(*prog, false);
    program_wrapper::run_graph_compilation(*prog);
    program_wrapper::post_optimize_graph(*prog, false);
    if (mode == pmd_mode::full) {
        program_wrapper::prepare_memory_dependencies(*prog);
    } else if (mode == pmd_mode::legacy_only) {
        for (auto* n : prog->get_processing_order()) {
            n->add_memory_dependency(*n);
        }
        program_wrapper::apply_opt_pass<basic_memory_dependencies>(*prog);
        program_wrapper::apply_opt_pass<skipped_branch_memory_dependencies>(*prog);
        program_wrapper::apply_opt_pass<oooq_memory_dependencies>(*prog);
    }
    return prog;
}

// ONLY VALID FOR GRAPHS WITHOUT stateless_kv: it re-implements the sum branch of HEAD basic_memory_dependencies and not its stateless_kv
// marking (HEAD bmd :49-56).
// Re-implementation of the HEAD sum walk of basic_memory_dependencies (before the defect B transition): for each in-place sum writer W
// with addend dependency E: W itself and the node reached by `root = E; while (root->cbo && has deps) root = root->dep(0)`.
std::set<std::string> head_sum_marks(program& prog) {
    std::set<std::string> marked;
    for (auto* node : prog.get_processing_order()) {
        if (node->get_preferred_impl_type() != impl_types::onednn) {
            continue;
        }
        const int32_t idx = onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(*node);
        if (idx == -1) {
            continue;
        }
        bool is_sum = false;
        for (const auto& f : node->get_fused_primitives()) {
            if (f.is_type<eltwise>() && f.deps.size() == 1 && f.has_outer_dep() && f.outer_dep_start_idx == idx &&
                f.typed_desc<eltwise>()->mode == eltwise_mode::sum) {
                is_sum = true;
                break;
            }
        }
        if (!is_sum) {
            continue;
        }
        marked.insert(node->id());
        auto* root = &node->get_dependency(static_cast<size_t>(idx));
        while (root->can_be_optimized() && !root->get_dependencies().empty()) {
            root = &root->get_dependency(0);
        }
        marked.insert(root->id());
    }
    return marked;
}

// can_share_buffer()==false set of a program, by node id
std::set<std::string> non_shareable_set(program& prog) {
    std::set<std::string> r;
    for (auto* n : prog.get_processing_order()) {
        if (!n->can_share_buffer()) {
            r.insert(n->id());
        }
    }
    return r;
}

// Defect B transition against HEAD: H = flags before bmd (pbf, defaults) + the HEAD sum walk; F = flags after the full PMD.
// Asserts H subset of F and F \ H == expected, every node of F \ H being a static cbo non-rs node (bypasses the pool anyway).
void expect_defect_b_flag_diff(engine& eng, const topology& t, const ExecutionConfig& cfg, const std::set<std::string>& expected, const std::string& tag) {
    auto base = build_to_pmd(eng, t, cfg, pmd_mode::none);
    std::set<std::string> head = non_shareable_set(*base);
    for (const auto& id : head_sum_marks(*base)) {
        head.insert(id);
    }
    auto full = build_to_pmd(eng, t, cfg, pmd_mode::full);
    const auto after = non_shareable_set(*full);
    for (const auto& id : head) {
        EXPECT_TRUE(after.count(id) != 0) << tag << ": HEAD-marked node " << id << " lost its mark";
    }
    std::set<std::string> added;
    for (const auto& id : after) {
        if (head.count(id) == 0) {
            added.insert(id);
        }
    }
    EXPECT_EQ(added, expected) << tag << ": flags added relative to HEAD\n" << dump_flags(*full);
    for (const auto& id : added) {
        const auto& n = full->get_node(id);
        EXPECT_TRUE(n.can_be_optimized() && !n.is_dynamic() && !n.is_runtime_skippable()) << tag << ": " << id << " would be newly excluded from the pool";
    }
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
// A1. conv -> reshape(cbo) -> reorder(cbo): one class, root is the producer, MUST_VIEW members, no fallback.
// ==============================================================================================================================
TEST(alias_analysis, A1_view_chain_one_class_root_is_producer) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, chain_topology(), {"O"});
    prog->get_node("R").can_be_optimized(true);
    prog->get_node("Q").can_be_optimized(true);
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "P");
    EXPECT_EQ(c, aa.class_of(prog->get_node("R")));
    EXPECT_EQ(c, aa.class_of(prog->get_node("Q")));
    EXPECT_NE(c, aa.class_of(prog->get_node("O")));
    EXPECT_EQ(c->root_node, &prog->get_node("P")) << dump_aa(aa);
    EXPECT_EQ(c->root.index, 0);
    ASSERT_EQ(c->members.size(), 3u) << dump_aa(aa);
    EXPECT_KIND(c, prog, "P", alias_kind::NONE, offset_tag::ZERO, alias_role::allocator);
    EXPECT_KIND(c, prog, "R", alias_kind::MUST_VIEW, offset_tag::ZERO, alias_role::zero_alloc_member);
    EXPECT_KIND(c, prog, "Q", alias_kind::MUST_VIEW, offset_tag::ZERO, alias_role::zero_alloc_member);
    EXPECT_EQ(c->fallback, 0u) << dump_aa(aa);
    EXPECT_EQ(c->domain, 0u);
    EXPECT_FALSE(c->has_fallback());
    EXPECT_EQ(c->get_placement(), placement::POOL);
    ASSERT_EQ(c->readers.size(), 1u);
    EXPECT_EQ(c->readers[0], &prog->get_node("O"));
    EXPECT_EQ(aa.root_of(aa.class_of(prog->get_node("Q"))->members.back().id), c->root);
}

// ==============================================================================================================================
// A2. static in-place concat: the holder is the root, predecessors are MUST_SUBREGION slots, no fallback.
// ==============================================================================================================================
TEST(alias_analysis, A2_static_concat_hand) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, concat_topology(), {"O"});
    prog->get_node("C").can_be_optimized(true);
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "C");
    EXPECT_EQ(c->root_node, &prog->get_node("C")) << dump_aa(aa);
    ASSERT_EQ(c->members.size(), 3u) << dump_aa(aa);
    EXPECT_KIND(c, prog, "C", alias_kind::NONE, offset_tag::ZERO, alias_role::allocator);
    EXPECT_KIND(c, prog, "P1", alias_kind::MUST_SUBREGION, offset_tag::LAYOUT_PADDING, alias_role::zero_alloc_member);
    EXPECT_KIND(c, prog, "P2", alias_kind::MUST_SUBREGION, offset_tag::LAYOUT_PADDING, alias_role::zero_alloc_member);
    EXPECT_EQ(c->fallback, 0u) << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::POOL);
    ASSERT_EQ(c->readers.size(), 1u);
    EXPECT_EQ(c->readers[0], &prog->get_node("O"));
}

// The same through the real passes: the wiring in prepare_memory_dependencies publishes the engine on the program.
TEST(alias_analysis, A2_static_concat_pipeline_and_program_wiring) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, pipeline_concat_topology(), make_cfg(eng, QueueTypes::in_order, true));
    const auto* aa_ptr = prog->get_alias_analysis();
    ASSERT_NE(aa_ptr, nullptr) << "prepare_memory_dependencies must publish the alias analysis";
    const auto& aa = *aa_ptr;
    ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::in_order);
    ASSERT_TRUE(prog->get_node("C").can_be_optimized()) << "precondition: C is an in-place concat\n" << dump_flags(*prog);
    GET_CLASS(c, aa, prog, "C");
    EXPECT_EQ(c->root_node, &prog->get_node("C")) << dump_aa(aa);
    EXPECT_KIND(c, prog, "P1", alias_kind::MUST_SUBREGION, offset_tag::LAYOUT_PADDING, alias_role::zero_alloc_member);
    EXPECT_KIND(c, prog, "P2", alias_kind::MUST_SUBREGION, offset_tag::LAYOUT_PADDING, alias_role::zero_alloc_member);
    EXPECT_EQ(c->fallback, 0u) << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::POOL);
}

// ==============================================================================================================================
// A3. cascaded concat: the outer holder is the root, the inner concat is a slot of it (a concat predecessor that is a concat is allowed).
// ==============================================================================================================================
TEST(alias_analysis, A3_cascaded_concat_root_is_outer) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", f32_l()));
    t.add(input_layout("in2", f32_l()));
    t.add(input_layout("in3", f32_l()));
    t.add(activation("P1", input_info("in1"), activation_func::relu));
    t.add(activation("P2", input_info("in2"), activation_func::abs));
    t.add(activation("P3", input_info("in3"), activation_func::relu));
    t.add(concatenation("C1", {input_info("P1"), input_info("P2")}, 1));
    t.add(concatenation("C2", {input_info("C1"), input_info("P3")}, 1));
    t.add(activation("O", input_info("C2"), activation_func::abs));
    auto prog = make_hand(eng, t, {"O"});
    prog->get_node("C1").can_be_optimized(true);
    prog->get_node("C2").can_be_optimized(true);
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "C2");
    EXPECT_EQ(c->root_node, &prog->get_node("C2")) << dump_aa(aa);
    ASSERT_EQ(c->members.size(), 5u) << dump_aa(aa);
    EXPECT_KIND(c, prog, "C1", alias_kind::MUST_SUBREGION, offset_tag::LAYOUT_PADDING, alias_role::zero_alloc_member);
    EXPECT_KIND(c, prog, "P1", alias_kind::MUST_SUBREGION, offset_tag::LAYOUT_PADDING, alias_role::zero_alloc_member);
    EXPECT_KIND(c, prog, "P3", alias_kind::MUST_SUBREGION, offset_tag::LAYOUT_PADDING, alias_role::zero_alloc_member);
    EXPECT_EQ(c->fallback, 0u) << "cascaded concat must not be a role conflict\n" << dump_aa(aa);
}

// ==============================================================================================================================
// A4. dynamic runtime-skippable concat / view: MAY kinds, RUNTIME offsets, role may_allocate_member.
// ==============================================================================================================================
TEST(alias_analysis, A4_dynamic_runtime_skippable_kinds_are_may) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}));
    t.add(input_layout("in2", layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}));
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(activation("R", input_info("C"), activation_func::abs));
    t.add(reorder("O", input_info("R"), format::bfyx, data_types::f32));
    auto prog = make_hand(eng, t, {"O"});
    ASSERT_TRUE(prog->get_node("C").is_dynamic());
    prog->get_node("C").can_be_optimized(true);
    prog->get_node("C").set_runtime_skippable(true);
    prog->get_node("R").set_runtime_skippable(true);
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "C");
    EXPECT_EQ(c->root_node, &prog->get_node("C")) << dump_aa(aa);
    EXPECT_KIND(c, prog, "P1", alias_kind::MAY_SUBREGION, offset_tag::RUNTIME, alias_role::may_allocate_member);
    EXPECT_KIND(c, prog, "P2", alias_kind::MAY_SUBREGION, offset_tag::RUNTIME, alias_role::may_allocate_member);
    EXPECT_KIND(c, prog, "R", alias_kind::MAY_VIEW, offset_tag::ZERO, alias_role::may_allocate_member);
    EXPECT_EQ(c->fallback & (fb_oooq | fb_role_conflict | fb_nonzero_port_root | fb_internal_net | fb_body_ctrl_flow | fb_shape_of), 0u)
        << dump_aa(aa);
}

// ==============================================================================================================================
// A5. crop (cbo) is a view of its producer: the producer is the root (the holder of a Crop is the producer).
// ==============================================================================================================================
TEST(alias_analysis, A5_crop_root_is_producer) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l(ov::PartialShape{1, 16, 8, 8})));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(crop("K", input_info("P"), tensor(1, 8, 8, 8), tensor(0, 0, 0, 0)));
    t.add(activation("O", input_info("K"), activation_func::abs));
    auto prog = make_hand(eng, t, {"O"});
    prog->get_node("K").can_be_optimized(true);
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "P");
    EXPECT_EQ(c, aa.class_of(prog->get_node("K")));
    EXPECT_EQ(c->root_node, &prog->get_node("P")) << dump_aa(aa);
    EXPECT_KIND(c, prog, "K", alias_kind::MUST_VIEW, offset_tag::STATIC_BYTES, alias_role::zero_alloc_member);
    EXPECT_EQ(c->fallback, 0u) << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::POOL);
}

// ==============================================================================================================================
// A8. fallback reasons and domain reasons, one by one.
// ==============================================================================================================================
TEST(alias_analysis, A8a_oooq_every_class_has_fb_oooq) {
    auto& eng = get_test_engine();
    auto prog_io = make_hand(eng, chain_topology(), {"O"}, QueueTypes::in_order);
    ASSERT_EQ(prog_io->get_config().get_queue_type(), QueueTypes::in_order);
    prog_io->get_node("R").can_be_optimized(true);
    auto aa_io = alias_analysis::build(*prog_io);
    for (const auto& c : aa_io.classes()) {
        EXPECT_EQ(c.fallback & fb_oooq, 0u) << "in-order must not report fb_oooq\n" << dump_aa(aa_io);
    }

    auto prog = make_hand(eng, chain_topology(), {"O"}, QueueTypes::out_of_order);
    ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::out_of_order) << "the OOOQ queue was not honoured";
    prog->get_node("R").can_be_optimized(true);
    auto aa = alias_analysis::build(*prog);
    ASSERT_FALSE(aa.classes().empty());
    for (const auto& c : aa.classes()) {
        EXPECT_NE(c.fallback & fb_oooq, 0u) << dump_aa(aa);
        EXPECT_EQ(c.get_placement(), placement::LEGACY);
    }
}

TEST(alias_analysis, A8b_cpu_member_static) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, chain_topology(), {"O"});
    prog->get_node("R").can_be_optimized(true);
    prog->get_node("Q").can_be_optimized(true);
    prog->get_node("Q").set_preferred_impl_type(impl_types::cpu);  // no selected impl: the preferred type decides
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "P");
    EXPECT_NE(c->fallback & fb_cpu_impl, 0u) << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::LEGACY);
}

TEST(alias_analysis, A8c_cpu_reader_static) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, chain_topology(), {"O"});
    prog->get_node("R").can_be_optimized(true);
    prog->get_node("Q").can_be_optimized(true);
    prog->get_node("O").set_preferred_impl_type(impl_types::cpu);
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "P");
    ASSERT_EQ(c->readers.size(), 1u);
    EXPECT_NE(c->fallback & fb_cpu_impl, 0u) << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::LEGACY);
}

TEST(alias_analysis, A8d_shape_of_reader_is_skipped_but_falls_back) {
    auto& eng = get_test_engine();
    topology t = chain_topology();
    t.add(shape_of("S", input_info("P"), data_types::i32));
    auto prog = make_hand(eng, t, {"O"});
    prog->get_node("R").can_be_optimized(true);
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "P");
    EXPECT_NE(c->fallback & fb_shape_of, 0u) << dump_aa(aa);
    for (const auto* r : c->readers) {
        EXPECT_NE(r->id(), "S") << "shape_of must not be a reader";
    }
    EXPECT_EQ(c->get_placement(), placement::LEGACY);
}

TEST(alias_analysis, A8e_output_member_is_user_output_domain) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, chain_topology(), {"O"});
    prog->get_node("R").can_be_optimized(true);
    prog->get_node("R").set_output(true);
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "P");
    EXPECT_NE(c->domain & dm_user_output, 0u) << dump_aa(aa);
    EXPECT_EQ(c->fallback, 0u) << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::OUTPUT_DOMAIN);
}

TEST(alias_analysis, A8f_mutable_data_propagated_output) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("D", input_info("in"), activation_func::relu));
    t.add(activation("X", input_info("D"), activation_func::abs));
    auto mem = eng.allocate_memory(f32_l());
    t.add(mutable_data("md", {input_info("D")}, mem));
    auto prog = make_hand(eng, t, {"X", "md"});
    ASSERT_FALSE(prog->get_node("D").is_output());
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "D");
    EXPECT_NE(c->domain & dm_user_output, 0u) << "D becomes an output through the mutable_data user\n" << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::OUTPUT_DOMAIN);
}

TEST(alias_analysis, A8g_input_layout_root_is_external) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("R", input_info("in"), activation_func::abs));
    t.add(activation("O", input_info("R"), activation_func::abs));
    auto prog = make_hand(eng, t, {"O"});
    prog->get_node("R").can_be_optimized(true);
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "in");
    EXPECT_EQ(c, aa.class_of(prog->get_node("R")));
    EXPECT_EQ(c->root_node, &prog->get_node("in")) << dump_aa(aa);
    EXPECT_NE(c->domain & dm_external_root, 0u);
    EXPECT_EQ(c->fallback, 0u) << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::EXTERNAL);
}

TEST(alias_analysis, A8h_read_value_and_assign_are_state) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(read_value{"rv", {input_info("in")}, "v0", {f32_l()}});
    t.add(activation("X", input_info("rv"), activation_func::abs));
    t.add(assign("as", {input_info("X")}, "v0", f32_l()));
    auto prog = make_hand(eng, t, {"as"});
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(crv, aa, prog, "rv");
    EXPECT_NE(crv->domain & dm_persistent, 0u) << dump_aa(aa);
    EXPECT_EQ(crv->domain & dm_external_root, 0u) << "a persistent root is STATE, not EXTERNAL";
    EXPECT_EQ(crv->get_placement(), placement::STATE) << dump_aa(aa);
    GET_CLASS(cas, aa, prog, "as");
    EXPECT_NE(cas->domain & dm_persistent, 0u);
    EXPECT_EQ(cas->get_placement(), placement::STATE) << dump_aa(aa);
}

TEST(alias_analysis, A8i_body_program_and_A8j_internal_program) {
    auto& eng = get_test_engine();
    {
        auto prog = make_hand(eng, chain_topology(), {"O"}, QueueTypes::in_order, /*internal=*/false, /*body=*/true);
        ASSERT_TRUE(prog->is_body_program());
        auto aa = alias_analysis::build(*prog);
        for (const auto& c : aa.classes()) {
            EXPECT_NE(c.fallback & fb_body_ctrl_flow, 0u) << dump_aa(aa);
            EXPECT_EQ(c.fallback & fb_internal_net, 0u);
        }
    }
    {
        auto prog = make_hand(eng, chain_topology(), {"O"}, QueueTypes::in_order, /*internal=*/true, /*body=*/false);
        ASSERT_TRUE(prog->is_internal_program());
        auto aa = alias_analysis::build(*prog);
        for (const auto& c : aa.classes()) {
            EXPECT_NE(c.fallback & fb_internal_net, 0u) << dump_aa(aa);
            EXPECT_EQ(c.fallback & fb_body_ctrl_flow, 0u);
        }
    }
}

// Dynamic nodes whose supported implementation set contains a CPU implementation are "maybe CPU" (has_impl_for probe); a static node
// never gets fb_maybe_cpu.
TEST(alias_analysis, A8k_dynamic_node_with_cpu_impl_is_maybe_cpu) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}));
    t.add(shape_of("E", input_info("in"), data_types::i32));
    t.add(reorder("O", input_info("E"), format::bfyx, data_types::f16));
    auto prog = make_hand(eng, t, {"O"});
    auto& E = prog->get_node("E");
    ASSERT_TRUE(E.is_dynamic());
    if (!E.type()->has_impl_for(E, impl_types::cpu, shape_types::dynamic_shape)) {
        GTEST_SKIP() << "no dynamic CPU implementation accepts this shape_of (has_impl_for == false), maybe-CPU not constructible here";
    }
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "E");
    EXPECT_NE(c->fallback & fb_maybe_cpu, 0u) << dump_aa(aa);
    EXPECT_EQ(c->fallback & fb_cpu_impl, 0u) << "a dynamic node is never a sure CPU node without a selected impl\n" << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::LEGACY);

    // static counterpart: no maybe-CPU probe result
    topology ts;
    ts.add(input_layout("in", f32_l()));
    ts.add(eltwise("E", {input_info("in"), input_info("in")}, eltwise_mode::sum));
    ts.add(activation("O", input_info("E"), activation_func::abs));
    auto progs = make_hand(eng, ts, {"O"});
    auto aas = alias_analysis::build(*progs);
    GET_CLASS(cs, aas, progs, "E");
    EXPECT_EQ(cs->fallback & (fb_maybe_cpu | fb_cpu_impl), 0u) << dump_aa(aas);
}

// A primitive whose only CPU impl is static (non_max_suppression) is still maybe-CPU as a dynamic node: the runtime factory picks a static
// impl per concrete shape. Not in a shape_of subgraph, so fb_maybe_cpu is the only reason.
TEST(alias_analysis, A8l_dynamic_node_with_static_only_cpu_impl_is_maybe_cpu) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("boxes", layout{ov::PartialShape{1, ov::Dimension::dynamic(), 4}, data_types::f32, format::bfyx}));
    t.add(input_layout("scores", layout{ov::PartialShape{1, 1, ov::Dimension::dynamic()}, data_types::f32, format::bfyx}));
    t.add(non_max_suppression("nms", input_info("boxes"), input_info("scores"), 1, false, true));
    t.add(reorder("O", input_info("nms"), format::bfyx, data_types::f32));
    std::shared_ptr<program> prog;
    try {
        prog = make_hand(eng, t, {"O"});
    } catch (const std::exception& e) {
        GTEST_SKIP() << "dynamic non_max_suppression hand graph cannot be built here: " << e.what();
    }
    auto& nms = prog->get_node("nms");
    if (!nms.is_dynamic() || !nms.type()->has_impl_for(nms, impl_types::cpu) ||
        nms.type()->has_impl_for(nms, impl_types::cpu, shape_types::dynamic_shape)) {
        GTEST_SKIP() << "premise not met: node dynamic=" << nms.is_dynamic() << ", cpu impl (any shape)=" << nms.type()->has_impl_for(nms, impl_types::cpu)
                     << ", dynamic cpu impl=" << nms.type()->has_impl_for(nms, impl_types::cpu, shape_types::dynamic_shape);
    }
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "nms");
    EXPECT_EQ(c->fallback, static_cast<uint32_t>(fb_maybe_cpu)) << dump_aa(aa);
}

// uid collision: two nodes with the same unique id (nodes created after compile_graph keep id 0 in real graphs).
TEST(alias_analysis, A17_duplicate_uid_identity_lookup_is_exact) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, chain_topology(), {"O"});
    prog->get_node("R").can_be_optimized(true);
    prog->get_node("O").set_unique_id(prog->get_node("P").get_unique_id());  // O collides with P
    auto aa = alias_analysis::build(*prog);
    EXPECT_GT(aa.duplicate_uid_count(), 0u);
    const alias_class* cp = aa.class_of(prog->get_node("P"));
    const alias_class* co = aa.class_of(prog->get_node("O"));
    ASSERT_NE(cp, nullptr);
    ASSERT_NE(co, nullptr);
    EXPECT_NE(cp, co);
    EXPECT_EQ(cp->root_node, &prog->get_node("P"));
    EXPECT_EQ(co->root_node, &prog->get_node("O"));
    EXPECT_EQ(co->members.size(), 1u);
    // uid lookup: the shared uid is ambiguous, so both key and root_of are not resolved; other uids resolve normally
    const uint32_t uid = static_cast<uint32_t>(prog->get_node("P").get_unique_id());
    const tensor_id shared{uid, tensor_space::OUTPUT, 0};
    EXPECT_EQ(aa.class_of(shared), nullptr);
    EXPECT_EQ(aa.root_of(shared), shared);
    const auto& R = prog->get_node("R");
    EXPECT_EQ(aa.class_of(tensor_id{static_cast<uint32_t>(R.get_unique_id()), tensor_space::OUTPUT, 0}), cp) << "unique uid must still resolve";
    expect_identity_lookup_exact(*prog, aa);
}

// A21 (P1-B-S). uid level view of the collision data: a uid shared by two nodes is ambiguous (the first and the later tensor), others are not.
TEST(alias_analysis, A21_uid_is_ambiguous_accessor) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, chain_topology(), {"O"});
    prog->get_node("R").can_be_optimized(true);
    {
        auto aa = alias_analysis::build(*prog);
        EXPECT_EQ(aa.ambiguous_uid_count(), 0u);
        for (auto* n : prog->get_processing_order()) {
            EXPECT_FALSE(aa.uid_is_ambiguous(static_cast<uint32_t>(n->get_unique_id()))) << n->id();
        }
    }
    prog->get_node("O").set_unique_id(prog->get_node("P").get_unique_id());  // O collides with P
    auto aa = alias_analysis::build(*prog);
    const uint32_t shared = static_cast<uint32_t>(prog->get_node("P").get_unique_id());
    EXPECT_TRUE(aa.uid_is_ambiguous(shared));
    EXPECT_EQ(aa.ambiguous_uid_count(), 1u);
    EXPECT_FALSE(aa.uid_is_ambiguous(static_cast<uint32_t>(prog->get_node("R").get_unique_id()))) << "unique uids stay unambiguous";
    EXPECT_FALSE(aa.uid_is_ambiguous(0xFFFFFFF0u)) << "unknown uid";
}

// kv_cache: EVERY kv_cache has an alias edge to its input 0 (user decision 2026-11-20): MUST_VIEW (settled-view edge) when cbo, MAY_INPLACE
// when not. The variable (read_value) class therefore contains the kv_cache and the readers of the kv_cache are readers of that class.
TEST(alias_analysis, A18_kv_cache_cbo_and_non_cbo) {
    auto& eng = get_test_engine();
    auto make = [&]() {
        topology t;
        t.add(input_layout("state", layout{ov::PartialShape{1, 2, 4, 8}, data_types::f16, format::bfyx}));
        t.add(read_value{"rv", {input_info("state")}, "v0", {layout{ov::PartialShape{1, 2, 4, 8}, data_types::f16, format::bfyx}}});
        t.add(input_layout("new_kv", layout{ov::PartialShape{1, 2, 1, 8}, data_types::f16, format::bfyx}));
        t.add(kv_cache("kv", {input_info("rv"), input_info("new_kv")}, ov::op::util::VariableInfo{ov::PartialShape{1, 2, 4, 8}, ov::element::f16, "v0"}, 2, 0,
                       false, false, false));
        t.add(activation("O", input_info("kv"), activation_func::abs));
        return t;
    };
    auto readers_contain = [](const alias_class& c, const program_node& n) {
        return std::find(c.readers.begin(), c.readers.end(), &n) != c.readers.end();
    };
    for (const bool cbo : {true, false}) {
        std::shared_ptr<program> prog;
        try {
            prog = make_hand(eng, make(), {"O"});
        } catch (const std::exception& e) {
            GTEST_SKIP() << "kv_cache hand graph cannot be built here: " << e.what();
        }
        prog->get_node("kv").can_be_optimized(cbo);
        auto aa = alias_analysis::build(*prog);
        GET_CLASS(c, aa, prog, "kv");
        const std::string tag = cbo ? "cbo" : "non-cbo";
        EXPECT_EQ(c, aa.class_of(prog->get_node("rv"))) << tag << "\n" << dump_aa(aa);
        EXPECT_EQ(c->root_node, &prog->get_node("rv")) << tag;
        if (cbo) {
            EXPECT_KIND(c, prog, "kv", alias_kind::MUST_VIEW, offset_tag::ZERO, alias_role::zero_alloc_member);
        } else {
            EXPECT_KIND(c, prog, "kv", alias_kind::MAY_INPLACE, offset_tag::ZERO, alias_role::may_allocate_member);
        }
        EXPECT_NE(c->domain & dm_persistent, 0u) << tag;
        // the user of the kv_cache reads the variable buffer through the alias: it is a reader of this class (the MAY envelope used to miss it)
        EXPECT_TRUE(readers_contain(*c, prog->get_node("O"))) << tag << "\n" << dump_aa(aa);
        // the appended output [1,2,5,8] is larger than the variable layout [1,2,4,8]: the size check makes the class a role conflict, and any
        // fallback beats the domain, so it stays LEGACY (and it is a state class either way). Exactly one provider edge: no extra conflict.
        EXPECT_EQ(c->fallback, static_cast<uint32_t>(fb_role_conflict)) << tag << "\n" << dump_aa(aa);
        EXPECT_EQ(c->get_placement(), placement::LEGACY) << tag;
    }
}

namespace {
// kv_cache graph helper. `state_dyn` makes the variable layouts dynamic in the sequence axis.
topology kv_topology(bool state_dyn, const std::string& rv_variable, const std::string& kv_variable, bool through_activation = false) {
    const ov::Dimension seq = state_dyn ? ov::Dimension::dynamic() : ov::Dimension(4);
    const ov::PartialShape state_shape{1, 2, seq, 8};
    topology t;
    t.add(input_layout("state", layout{state_shape, data_types::f16, format::bfyx}));
    t.add(read_value{"rv", {input_info("state")}, rv_variable, {layout{state_shape, data_types::f16, format::bfyx}}});
    t.add(input_layout("new_kv", layout{ov::PartialShape{1, 2, state_dyn ? ov::Dimension::dynamic() : ov::Dimension(1), 8}, data_types::f16, format::bfyx}));
    std::string past = "rv";
    if (through_activation) {
        t.add(activation("mid", input_info("rv"), activation_func::abs));  // a non-view node between the variable and the kv_cache
        past = "mid";
    }
    t.add(kv_cache("kv", {input_info(past), input_info("new_kv")}, ov::op::util::VariableInfo{state_shape, ov::element::f16, kv_variable}, 2, 0, false, false, false));
    t.add(activation("O", input_info("kv"), activation_func::abs));
    return t;
}
}  // namespace

// K4: dynamic kv_cache. Sizes are unknown (no size check) and a read_value root is a persistent type (not an external root), so the kv class
// has no role / size / port fallback. FINDING: the class is nevertheless LEGACY in practice, because read_value only has CPU implementations
// (impls/cpu/read_value.cpp registers static and dynamic CPU impls only): the dynamic read_value is "maybe CPU" (fb_maybe_cpu), and a static
// one with a selected impl is fb_cpu_impl. The STATE placement of a kv class is only reachable if the CPU rule is lifted; war_safe (0.5b / 1a)
// must therefore not assume that a STATE class is the only shape of a kv class, and must treat both STATE and CPU-fallback kv classes as
// external-root / persistent-writer classes or return Unknown.
TEST(alias_analysis, A18d_dynamic_kv_cache_class_is_persistent_and_only_maybe_cpu) {
    auto& eng = get_test_engine();
    std::shared_ptr<program> prog;
    try {
        prog = make_hand(eng, kv_topology(true, "v0", "v0"), {"O"});
    } catch (const std::exception& e) {
        GTEST_SKIP() << "dynamic kv_cache hand graph cannot be built here: " << e.what();
    }
    ASSERT_TRUE(prog->get_node("kv").is_dynamic());
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "kv");
    EXPECT_EQ(c, aa.class_of(prog->get_node("rv"))) << dump_aa(aa);
    EXPECT_KIND(c, prog, "kv", alias_kind::MAY_INPLACE, offset_tag::ZERO, alias_role::may_allocate_member);
    EXPECT_NE(c->domain & dm_persistent, 0u);
    EXPECT_EQ(c->domain & dm_external_root, 0u);
    EXPECT_EQ(c->fallback, static_cast<uint32_t>(fb_maybe_cpu)) << "the only reason is the CPU-only read_value\n" << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::LEGACY) << dump_aa(aa);
}

// K1 negative: dependency 0 does not resolve to the read_value of the kv_cache's variable -> no in-place edge, class is a role conflict.
TEST(alias_analysis, A18e_kv_cache_dep0_not_the_variable_read_value) {
    auto& eng = get_test_engine();
    for (const bool through_non_view : {true, false}) {
        std::shared_ptr<program> prog;
        try {
            // either a non-view node sits between the read_value and the kv_cache, or the read_value belongs to another variable
            prog = make_hand(eng, through_non_view ? kv_topology(false, "v0", "v0", true) : kv_topology(false, "other", "v0"), {"O"});
        } catch (const std::exception& e) {
            GTEST_SKIP() << "kv_cache hand graph cannot be built here: " << e.what();
        }
        auto aa = alias_analysis::build(*prog);
        GET_CLASS(c, aa, prog, "kv");
        const std::string tag = through_non_view ? "non-view node" : "other variable_id";
        EXPECT_NE(c, aa.class_of(prog->get_node("rv"))) << tag << ": no alias edge may be built\n" << dump_aa(aa);
        EXPECT_NE(c->fallback & fb_role_conflict, 0u) << tag << "\n" << dump_aa(aa);
        EXPECT_EQ(c->get_placement(), placement::LEGACY) << tag;
    }
}

// K2: a compressed kv_cache has state-memory outputs that the analysis does not model: all its ports are role conflicts.
TEST(alias_analysis, A18f_compressed_kv_cache_is_conservative_fallback) {
    auto& eng = get_test_engine();
    std::shared_ptr<program> prog;
    try {
        auto t = kv_topology(false, "v0", "v0");
        prog = make_hand(eng, t, {"O"});
        const_cast<kv_cache&>(*std::static_pointer_cast<const kv_cache>(prog->get_node("kv").get_primitive())).compressed = true;
    } catch (const std::exception& e) {
        GTEST_SKIP() << "kv_cache hand graph cannot be built here: " << e.what();
    }
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "kv");
    EXPECT_NE(c->fallback & fb_role_conflict, 0u) << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::LEGACY);
}

// K3: a kv_cache that is an in-place concat predecessor AND the in-place target of a variable: the slot provider must not hide the kv edge.
TEST(alias_analysis, A18g_kv_cache_as_concat_slot_still_joins_variable_class) {
    auto& eng = get_test_engine();
    topology t = kv_topology(false, "v0", "v0");
    t.add(input_layout("other", layout{ov::PartialShape{1, 2, 5, 8}, data_types::f16, format::bfyx}));
    t.add(concatenation("C", {input_info("kv"), input_info("other")}, 2));
    t.add(activation("O2", input_info("C"), activation_func::abs));
    std::shared_ptr<program> prog;
    try {
        prog = make_hand(eng, t, {"O", "O2"});
    } catch (const std::exception& e) {
        GTEST_SKIP() << "graph cannot be built here: " << e.what();
    }
    prog->get_node("C").can_be_optimized(true);
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "kv");
    EXPECT_EQ(c, aa.class_of(prog->get_node("rv"))) << "the variable alias must not be lost\n" << dump_aa(aa);
    EXPECT_EQ(c, aa.class_of(prog->get_node("C"))) << dump_aa(aa);
    EXPECT_NE(c->fallback & fb_role_conflict, 0u) << "two allocators (variable and concat holder)\n" << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::LEGACY);
}

// stateless_kv: dep 0 may be updated in place at runtime: MAY_INPLACE edge, class has the persistent domain.
TEST(alias_analysis, A19_stateless_kv_edge_to_past) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("past", layout{ov::PartialShape{1, 2, -1, 4}, data_types::f32, format::bfyx}));
    t.add(input_layout("new_token", layout{ov::PartialShape{1, 2, 1, 4}, data_types::f32, format::bfyx}));
    t.add(input_layout("present_len", layout{ov::PartialShape{1}, data_types::i64, format::bfyx}));
    auto kv = stateless_kv("stateless_kv", {input_info("past"), input_info("new_token"), input_info("present_len")}, 2, true);
    kv.num_outputs = 2;
    kv.output_data_types = {data_types::f32, data_types::f32};
    t.add(kv);
    t.add(reorder("result", input_info("stateless_kv", 0), format::bfyx, data_types::f32));
    std::shared_ptr<program> prog;
    try {
        prog = make_hand(eng, t, {"result"});
    } catch (const std::exception& e) {
        GTEST_SKIP() << "stateless_kv hand graph cannot be built here: " << e.what();
    }
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "past");
    EXPECT_EQ(c, aa.class_of(prog->get_node("stateless_kv"))) << dump_aa(aa);
    EXPECT_KIND(c, prog, "stateless_kv", alias_kind::MAY_INPLACE, offset_tag::ZERO, alias_role::may_allocate_member);
    EXPECT_NE(c->domain & dm_persistent, 0u);
    EXPECT_NE(c->domain & dm_external_root, 0u) << "past is an input_layout";
    // K2: this node has two outputs; _outputs[1] is a reinterpret of the present buffer decided at runtime (primitive_inst.cpp:~1088), which
    // the analysis does not model: all ports are fb_role_conflict until 0.5b, so the class falls back.
    EXPECT_NE(c->fallback & fb_role_conflict, 0u) << dump_aa(aa);
    const alias_class* c1 = aa.class_of(prog->get_node("stateless_kv"), 1);
    ASSERT_NE(c1, nullptr);
    EXPECT_NE(c1->fallback & fb_role_conflict, 0u) << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::LEGACY);
}

// lora: opaque alias with its input 0 -> fb_role_conflict (never promoted).
TEST(alias_analysis, A20_lora_is_opaque_role_conflict) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("main", layout{ov::PartialShape{1, 1, 256}, data_types::f32, format::bfyx}));
    t.add(input_layout("input", layout{ov::PartialShape{1, 1, 128}, data_types::f32, format::bfyx}));
    t.add(input_layout("a", layout{ov::PartialShape{16, 128}, data_types::f32, format::bfyx}));
    t.add(input_layout("alpha", layout{ov::PartialShape{1, 16}, data_types::f32, format::bfyx}));
    t.add(input_layout("b", layout{ov::PartialShape{256, 16}, data_types::f32, format::bfyx}));
    t.add(activation("fc_like", input_info("main"), activation_func::relu));
    t.add(lora("lora", {input_info("fc_like"), input_info("input"), input_info("a"), input_info("alpha"), input_info("b")}, true));
    t.add(activation("O", input_info("lora"), activation_func::abs));
    std::shared_ptr<program> prog;
    try {
        prog = make_hand(eng, t, {"O"});
    } catch (const std::exception& e) {
        GTEST_SKIP() << "lora hand graph cannot be built here: " << e.what();
    }
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "lora");
    EXPECT_EQ(c, aa.class_of(prog->get_node("fc_like"))) << dump_aa(aa);
    EXPECT_NE(c->fallback & fb_role_conflict, 0u) << dump_aa(aa);
    EXPECT_KIND(c, prog, "lora", alias_kind::MAY_INPLACE, offset_tag::ZERO, alias_role::may_allocate_member);
    EXPECT_EQ(c->get_placement(), placement::LEGACY);
}

// ==============================================================================================================================
// A9. a view of port 1 of a multi-output node: its owner is the tensor (n, 1), which is not a promotable root yet.
// ==============================================================================================================================
TEST(alias_analysis, A9_view_of_nonzero_port_has_nonzero_port_root) {
    auto& eng = get_test_engine();
    const int top_k = 2;
    auto input_mem = eng.allocate_memory({data_types::f32, format::bfyx, {2, 4, 2, 2}});
    auto top_k_input = eng.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}});
    topology t;
    t.add(input_layout("input", input_mem->get_layout()));
    t.add(cldnn::data("const", {top_k_input}));
    auto arg = arg_max_min("arg_max", {input_info("input"), input_info("const")}, ov::op::TopKMode::MAX, top_k, 0, ov::op::TopKSortType::SORT_VALUES,
                           false, false, data_types::f32, 2);
    arg.output_paddings = {padding(), padding()};
    arg.output_data_types = {optional_data_type{data_types::f32}, optional_data_type{data_types::f32}};
    t.add(arg);
    t.add(activation("R", input_info("arg_max", 1), activation_func::abs));
    t.add(activation("O", input_info("R"), activation_func::abs));
    auto prog = make_hand(eng, t, {"O"});
    ASSERT_EQ(prog->get_node("arg_max").get_outputs_count(), 2u);
    prog->get_node("R").can_be_optimized(true);
    auto aa = alias_analysis::build(*prog);
    const alias_class* c1 = aa.class_of(prog->get_node("arg_max"), 1);
    ASSERT_NE(c1, nullptr) << dump_aa(aa);
    EXPECT_EQ(c1, aa.class_of(prog->get_node("R"))) << dump_aa(aa);
    EXPECT_EQ(c1->root_node, &prog->get_node("arg_max")) << dump_aa(aa);
    EXPECT_EQ(c1->root.index, 1);
    EXPECT_NE(c1->fallback & fb_nonzero_port_root, 0u) << dump_aa(aa);
    EXPECT_EQ(c1->get_placement(), placement::LEGACY);
    const alias_class* c0 = aa.class_of(prog->get_node("arg_max"), 0);
    ASSERT_NE(c0, nullptr);
    EXPECT_NE(c0, c1) << "ports of one node are separate tensors";
    EXPECT_EQ(c0->fallback & fb_nonzero_port_root, 0u);
    // the owner walk of the helper reports the same port
    const auto owner = program_helpers::resolve_buffer_owner(prog->get_node("R"), program_helpers::view_resolution::settled_only);
    EXPECT_EQ(owner.owner, &prog->get_node("arg_max"));
    EXPECT_EQ(owner.owner_port, 1);
}

// ==============================================================================================================================
// A10. cache export / import: the lazily built analysis of the imported program equals the one of the original.
// ==============================================================================================================================
TEST(alias_analysis, A10_import_builds_the_same_classes) {
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng, QueueTypes::in_order, true);
    auto prog = program::build_program(eng, pipeline_concat_topology(), cfg);
    ASSERT_TRUE(prog->get_node("C").can_be_optimized()) << dump_flags(*prog);
    program::ptr imported;
    ASSERT_NO_THROW(imported = roundtrip(eng, *prog, cfg));
    ASSERT_NE(imported, nullptr);
    EXPECT_EQ(imported->get_alias_analysis(), nullptr) << "import does not run prepare_memory_dependencies";
    auto a = alias_analysis::build(*prog);
    auto b = alias_analysis::build(*imported);
    EXPECT_EQ(dump_aa(a), dump_aa(b));
    ASSERT_EQ(a.classes().size(), b.classes().size());
}

// ==============================================================================================================================
// A11. dry_run: prepare_memory_dependencies is skipped, so is the analysis.
// ==============================================================================================================================
TEST(alias_analysis, A11_dry_run_does_not_build) {
#ifdef GPU_DEBUG_CONFIG
    auto& eng = get_test_engine();
    auto normal = program::build_program(eng, pipeline_concat_topology(), make_cfg(eng, QueueTypes::in_order, true));
    ASSERT_NE(normal->get_alias_analysis(), nullptr);
    auto cfg = make_cfg(eng, QueueTypes::in_order, true);
    cfg.set_user_property({ov::intel_gpu::dry_run_path("/tmp/alias_analysis_dry_run")});
    ASSERT_FALSE(cfg.get_dry_run_path().empty());
    auto dry = program::build_program(eng, pipeline_concat_topology(), cfg);
    EXPECT_EQ(dry->get_alias_analysis(), nullptr);
#else
    GTEST_SKIP() << "dry_run is a debug-caps option (GPU_DEBUG_CONFIG)";
#endif
}

// ==============================================================================================================================
// A12. building twice is deterministic; members and readers are ordered by the processing order.
// ==============================================================================================================================
TEST(alias_analysis, A12_deterministic_and_po_ordered) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, pipeline_concat_topology(), make_cfg(eng, QueueTypes::in_order, true));
    auto a = alias_analysis::build(*prog);
    auto b = alias_analysis::build(*prog);
    EXPECT_EQ(dump_aa(a), dump_aa(b));
    const auto& po = prog->get_processing_order();
    for (const auto& c : a.classes()) {
        for (size_t i = 1; i < c.members.size(); ++i) {
            const auto pa = po.get_processing_number(const_cast<program_node*>(c.members[i - 1].node));
            const auto pb = po.get_processing_number(const_cast<program_node*>(c.members[i].node));
            EXPECT_LE(pa, pb) << "members out of PO order\n" << dump_aa(a);
        }
        for (size_t i = 1; i < c.readers.size(); ++i) {
            EXPECT_LT(po.get_processing_number(const_cast<program_node*>(c.readers[i - 1])),
                      po.get_processing_number(const_cast<program_node*>(c.readers[i])))
                << "readers out of PO order\n"
                << dump_aa(a);
        }
    }
    // every tensor is in exactly the class that lists it
    for (const auto& c : a.classes()) {
        for (const auto& m : c.members) {
            EXPECT_EQ(a.class_of(*m.node, m.id.index), &c);
        }
    }
    for (auto* n : po) {
        EXPECT_NE(a.class_of(*n), nullptr) << n->id();
    }
}

// ==============================================================================================================================
// A13. recursion consistency: for a class without fallback, every reader r of a member has a restriction that meets the members.
// ==============================================================================================================================
namespace {
void expect_recursion_consistent(program& prog, const alias_analysis& aa) {
    size_t checked = 0;
    for (const auto& c : aa.classes()) {
        if (c.has_fallback()) {
            continue;
        }
        std::set<uint32_t> member_uids;
        for (const auto& m : c.members) {
            member_uids.insert(static_cast<uint32_t>(m.node->get_unique_id()));
        }
        for (const auto* r : c.readers) {
            const auto& restr = r->get_memory_dependencies();
            const bool meets = std::any_of(restr.begin(), restr.end(), [&](uint32_t u) {
                return member_uids.count(u) != 0;
            });
            EXPECT_TRUE(meets) << "reader " << r->id() << " of the class rooted at " << c.root_node->id() << " has no restriction on any member\n"
                               << dump_aa(aa);
            ++checked;
        }
    }
    EXPECT_GT(checked, 0u);
    (void)prog;
}
}  // namespace

TEST(alias_analysis, A13_recursion_consistency_pipeline) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, pipeline_concat_topology(), make_cfg(eng, QueueTypes::in_order, true));
    ASSERT_NE(prog->get_alias_analysis(), nullptr);
    expect_recursion_consistent(*prog, *prog->get_alias_analysis());
}

// ==============================================================================================================================
// A14. alias_analysis::build is a pure read: prepare_memory_dependencies with the engine call produces exactly the restrictions and
// flags that the three dependency passes alone produce (both runs use the current bmd; this says nothing about HEAD, see A6d).
// ==============================================================================================================================
TEST(alias_analysis, A14_build_is_pure_read_pipeline) {
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng, QueueTypes::in_order, true);
    auto legacy = build_to_pmd(eng, pipeline_concat_topology(), cfg, pmd_mode::legacy_only);
    auto full = build_to_pmd(eng, pipeline_concat_topology(), cfg, pmd_mode::full);
    EXPECT_EQ(legacy->get_alias_analysis(), nullptr);
    ASSERT_NE(full->get_alias_analysis(), nullptr);
    const auto sl = take_snapshot(*legacy);
    const auto sf = take_snapshot(*full);
    EXPECT_TRUE(sl == sf) << diff_snapshot(sl, sf);
    // building the engine again afterwards changes nothing either
    alias_analysis::build(*full);
    EXPECT_TRUE(take_snapshot(*full) == sf);
}

// ==============================================================================================================================
// A15. a member that does not fit into the root allocation makes the class a role conflict.
// ==============================================================================================================================
TEST(alias_analysis, A15_member_larger_than_root_is_role_conflict) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", layout{ov::PartialShape{1, 4, 8, 8}, data_types::f16, format::bfyx}));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(reorder("R", input_info("P"), format::bfyx, data_types::f32));  // twice the bytes of P
    t.add(activation("O", input_info("R"), activation_func::abs));
    auto prog = make_hand(eng, t, {"O"});
    prog->get_node("R").can_be_optimized(true);
    auto aa = alias_analysis::build(*prog);
    GET_CLASS(c, aa, prog, "P");
    EXPECT_EQ(c, aa.class_of(prog->get_node("R")));
    EXPECT_NE(c->fallback & fb_role_conflict, 0u) << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::LEGACY);
}

// ==============================================================================================================================
// A16. C-P2-1 lock: an output reorder that is a view of its producer. The producer class is a user-output class (never promoted), and
// the producer and the unrelated nodes after it restrict each other (guaranteed today by oooq, which P2 must keep or replace by an
// infinite death).
// ==============================================================================================================================
TEST(alias_analysis, A16_output_view_lock) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(reorder("R", input_info("P"), format::bfyx, data_types::f32));
    t.add(input_layout("in2", f32_l()));
    t.add(activation("Z", input_info("in2"), activation_func::relu));
    t.add(activation("Y", input_info("Z"), activation_func::abs));
    auto prog = make_hand(eng, t, {"R", "Y"});
    prog->get_node("R").can_be_optimized(true);
    ASSERT_TRUE(prog->get_node("R").is_output());
    // the unrelated branch runs after the output view (otherwise the check below is vacuous)
    force_order(*prog, {"in", "P", "R", "in2", "Z", "Y"});
    // the same stages as prepare_memory_dependencies
    program_wrapper::prepare_memory_dependencies(*prog);
    ASSERT_NE(prog->get_alias_analysis(), nullptr);
    const auto& aa = *prog->get_alias_analysis();
    GET_CLASS(c, aa, prog, "P");
    EXPECT_EQ(c, aa.class_of(prog->get_node("R")));
    EXPECT_NE(c->domain & dm_user_output, 0u) << dump_aa(aa);
    EXPECT_EQ(c->get_placement(), placement::OUTPUT_DOMAIN) << "never POOL";
    const auto& po = prog->get_processing_order();
    auto& P = prog->get_node("P");
    auto& R = prog->get_node("R");
    auto contains = [](const program_node& n, const program_node& other) {
        const auto& r = n.get_memory_dependencies();
        return std::find(r.begin(), r.end(), static_cast<uint32_t>(other.get_unique_id())) != r.end();
    };
    size_t checked = 0;
    for (const char* id : {"Z", "Y"}) {
        auto& X = prog->get_node(id);
        if (po.get_processing_number(&X) < po.get_processing_number(&R)) {
            continue;  // only nodes after the output view
        }
        EXPECT_TRUE(contains(P, X)) << "producer must restrict " << id << "\n" << dump_flags(*prog);
        EXPECT_TRUE(contains(X, P)) << id << " must restrict the producer\n" << dump_flags(*prog);
        ++checked;
    }
    EXPECT_EQ(checked, 2u) << "vacuity: Z and Y must come after the output view";
}

// ==============================================================================================================================
// Engine API contract
// ==============================================================================================================================
TEST(alias_analysis, api_class_of_rejects_negative_port_and_unknown_nodes) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, chain_topology(), {"O"});
    auto aa = alias_analysis::build(*prog);
    EXPECT_ANY_THROW(aa.class_of(prog->get_node("P"), -1));
    EXPECT_EQ(aa.class_of(tensor_id{0xFFFFFFFFu, tensor_space::OUTPUT, 0}), nullptr);
    EXPECT_EQ(aa.class_of(prog->get_node("P"), 5), nullptr);
}

#ifdef ENABLE_ONEDNN_FOR_GPU

namespace {

#define AA_SKIP_IF_NO_IMMAD()                                                               \
    do {                                                                                    \
        if (!get_test_engine().get_device_info().supports_immad) {                          \
            GTEST_SKIP() << "the oneDNN in-place writer needs a oneDNN device (supports_immad)"; \
        }                                                                                   \
    } while (0)

const ov::PartialShape kFc{32, 16};

memory::ptr make_weights(engine& eng, int64_t o, int64_t i, int variant) {
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

ExecutionConfig opt_cfg(engine& eng, const ov::intel_gpu::ImplForcingMap& forced) {
    ExecutionConfig config = make_cfg(eng, QueueTypes::in_order, true);
    config.set_property(ov::intel_gpu::force_implementations(forced));
    return config;
}

// X = FC(in3) ocl; W = FC(in1) onednn; S = sum(W, X) [fused into W]; Z = FC(S) ocl; out
topology sum_topology(engine& eng, ov::intel_gpu::ImplForcingMap& forced) {
    topology t;
    t.add(input_layout("in1", layout{kFc, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kFc, data_types::f16, format::bfyx}));
    t.add(data("wX", make_weights(eng, 16, 16, 1)));
    t.add(fully_connected("X", input_info("in3"), "wX"));
    t.add(data("wW", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("S", {input_info("W"), input_info("X")}, eltwise_mode::sum));
    t.add(data("wZ", make_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("S"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    forced.emplace("X", impl_desc(impl_types::ocl));
    forced.emplace("W", impl_desc(impl_types::onednn));
    forced.emplace("Z", impl_desc(impl_types::ocl));
    return t;
}

}  // namespace

// ==============================================================================================================================
// A6. oneDNN sum: the writer is a MUST_INPLACE member of the class of its addend; the chain root is the first allocator.
// ==============================================================================================================================
TEST(alias_analysis, A6a_sum_writer_joins_addend_class) {
    AA_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    ov::intel_gpu::ImplForcingMap forced;
    auto prog = program::build_program(eng, sum_topology(eng, forced), opt_cfg(eng, forced));
    ASSERT_NE(prog->get_alias_analysis(), nullptr);
    const auto& aa = *prog->get_alias_analysis();
    ASSERT_NE(onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(prog->get_node("W")), -1) << "precondition: W carries the in-place sum\n"
                                                                                                << dump_flags(*prog);
    // Finding (A6b/A6c): the uid projection is not injective in real graphs. The engine looks nodes up by identity and reports the number
    // of collisions for the uid-keyed consumers.
    std::cout << "[A6a] duplicate_uid_count=" << aa.duplicate_uid_count() << std::endl;
    expect_identity_lookup_exact(*prog, aa);
    GET_CLASS(c, aa, prog, "X");
    EXPECT_EQ(c, aa.class_of(prog->get_node("W"))) << dump_aa(aa);
    EXPECT_EQ(c->root_node, &prog->get_node("X")) << dump_aa(aa);
    EXPECT_KIND(c, prog, "X", alias_kind::NONE, offset_tag::ZERO, alias_role::allocator);
    EXPECT_KIND(c, prog, "W", alias_kind::MUST_INPLACE, offset_tag::ZERO, alias_role::zero_alloc_member);
    EXPECT_FALSE(prog->get_node("X").can_share_buffer()) << "legacy marking of the chain root";
    EXPECT_FALSE(prog->get_node("W").can_share_buffer());
}

// Sum chain: W2 adds W1 (which already writes into X). One class, root = first allocator X, both writers MUST_INPLACE.
TEST(alias_analysis, A6b_sum_chain_root_is_first_allocator) {
    AA_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    ov::intel_gpu::ImplForcingMap forced;
    topology t;
    t.add(input_layout("in1", layout{kFc, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kFc, data_types::f16, format::bfyx}));
    t.add(data("wX", make_weights(eng, 16, 16, 1)));
    t.add(fully_connected("X", input_info("in3"), "wX"));
    t.add(data("w1", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W1", input_info("in1"), "w1"));
    t.add(eltwise("S1", {input_info("W1"), input_info("X")}, eltwise_mode::sum));
    t.add(activation("A", input_info("S1"), activation_func::relu));  // W2 runs after S1 (residual block shape)
    t.add(data("w2", make_weights(eng, 16, 16, 3)));
    t.add(fully_connected("W2", input_info("A"), "w2"));
    t.add(eltwise("S2", {input_info("W2"), input_info("S1")}, eltwise_mode::sum));
    t.add(data("wZ", make_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("S2"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    forced.emplace("X", impl_desc(impl_types::ocl));
    forced.emplace("W1", impl_desc(impl_types::onednn));
    forced.emplace("W2", impl_desc(impl_types::onednn));
    forced.emplace("Z", impl_desc(impl_types::ocl));
    auto prog = program::build_program(eng, t, opt_cfg(eng, forced));
    ASSERT_NE(prog->get_alias_analysis(), nullptr);
    const auto& aa = *prog->get_alias_analysis();
    ASSERT_NE(onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(prog->get_node("W1")), -1) << dump_flags(*prog);
    ASSERT_NE(onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(prog->get_node("W2")), -1)
        << "precondition: the chain of two sums must be formed\n"
        << dump_flags(*prog);
    std::cout << "[A6b] duplicate_uid_count=" << aa.duplicate_uid_count() << std::endl;
    expect_identity_lookup_exact(*prog, aa);
    GET_CLASS(c, aa, prog, "X");
    EXPECT_EQ(c, aa.class_of(prog->get_node("W1"))) << dump_aa(aa);
    EXPECT_EQ(c, aa.class_of(prog->get_node("W2"))) << dump_aa(aa);
    EXPECT_EQ(c->root_node, &prog->get_node("X")) << dump_aa(aa);
    EXPECT_KIND(c, prog, "W1", alias_kind::MUST_INPLACE, offset_tag::ZERO, alias_role::zero_alloc_member);
    EXPECT_KIND(c, prog, "W2", alias_kind::MUST_INPLACE, offset_tag::ZERO, alias_role::zero_alloc_member);
}

namespace {
topology sum_concat_topology(engine& eng, ov::intel_gpu::ImplForcingMap& forced) {
    topology t;
    t.add(input_layout("in1", layout{ov::PartialShape{1, 16, 32, 32}, data_types::f16, format::bfyx}));
    t.add(input_layout("in2", layout{ov::PartialShape{1, 32, 32, 32}, data_types::f16, format::bfyx}));
    auto conv_w = [&](size_t o, size_t i, int variant) {
        auto mem = eng.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(o), static_cast<int64_t>(i), 1, 1}, data_types::f16, format::bfyx});
        std::vector<ov::float16> v(o * i);
        for (size_t k = 0; k < v.size(); k++) {
            v[k] = ov::float16(static_cast<float>((k * (5 + variant)) % 7) * 0.03125f);
        }
        set_values(mem, v);
        return mem;
    };
    t.add(data("w_pq", conv_w(16, 16, 1)));
    t.add(convolution("P", input_info("in1"), "w_pq", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    t.add(convolution("Q", input_info("in1"), "w_pq", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    t.add(concatenation("C", {input_info("P"), input_info("Q")}, 1));
    t.add(data("w_w", conv_w(32, 32, 3)));
    t.add(convolution("W", input_info("in2"), "w_w", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    t.add(eltwise("S", {input_info("W"), input_info("C")}, eltwise_mode::sum));
    t.add(reorder("out_S", input_info("S"), format::bfyx, data_types::f32));
    forced = ov::intel_gpu::ImplForcingMap{{"P", ov::intel_gpu::ImplementationDesc{format::b_fs_yx_fsv16, "", impl_types::ocl}},
                                         {"Q", ov::intel_gpu::ImplementationDesc{format::b_fs_yx_fsv16, "", impl_types::ocl}},
                                         {"C", ov::intel_gpu::ImplementationDesc{format::b_fs_yx_fsv16, "", impl_types::ocl}},
                                         {"W", ov::intel_gpu::ImplementationDesc{format::b_fs_yx_fsv16, "", impl_types::onednn}}};
    return t;
}
}  // namespace

// Defect B: a sum whose addend is an in-place concat. The legacy walk steps from the holder to its first predecessor; the owner by edge
// direction is the holder. Both are marked (transition): holder and pred0 end up non-shareable; the class root is the holder.
TEST(alias_analysis, A6c_sum_into_concat_holder_defect_b) {
    AA_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    ov::intel_gpu::ImplForcingMap forced;
    auto t = sum_concat_topology(eng, forced);
    auto prog = program::build_program(eng, t, opt_cfg(eng, forced));
    ASSERT_NE(prog->get_alias_analysis(), nullptr);
    const auto& aa = *prog->get_alias_analysis();
    ASSERT_TRUE(prog->get_node("C").can_be_optimized()) << dump_flags(*prog);
    ASSERT_NE(onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(prog->get_node("W")), -1) << "precondition: sum into the holder\n"
                                                                                                << dump_flags(*prog);
    std::cout << "[A6c] duplicate_uid_count=" << aa.duplicate_uid_count() << std::endl;
    expect_identity_lookup_exact(*prog, aa);
    GET_CLASS(c, aa, prog, "C");
    EXPECT_EQ(c->root_node, &prog->get_node("C")) << dump_aa(aa);
    EXPECT_KIND(c, prog, "W", alias_kind::MUST_INPLACE, offset_tag::ZERO, alias_role::zero_alloc_member);
    EXPECT_KIND(c, prog, "P", alias_kind::MUST_SUBREGION, offset_tag::LAYOUT_PADDING, alias_role::zero_alloc_member);
    EXPECT_FALSE(prog->get_node("C").can_share_buffer()) << "defect B: the holder is the real allocator and is marked now\n" << dump_flags(*prog);
    EXPECT_FALSE(prog->get_node("P").can_share_buffer()) << "the legacy mark (and pbf) on the first predecessor stays\n" << dump_flags(*prog);
}

// (HEAD comparison via head_sum_marks: valid only for graphs without stateless_kv.)
// Defect B transition on the sum-into-concat graph, against the HEAD walk re-implemented in the test. HEAD marks the writer W and walks from
// the holder C to its first predecessor P (cbo concat steps to dep(0)); it does NOT mark C. The transition adds exactly C. The flag
// flips true -> false (it is serialized into the blob), but the allocation does not change: C is a static cbo node, which allocate_output
// already allocates directly. Restrictions are untouched by the transition (checked against the dependency passes of this build).
TEST(alias_analysis, A6d_defect_b_flag_flip_vs_head_walk_is_exactly_the_holder) {
    AA_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    ov::intel_gpu::ImplForcingMap forced;
    auto t = sum_concat_topology(eng, forced);
    auto cfg = opt_cfg(eng, forced);
    {
        auto base = build_to_pmd(eng, t, cfg, pmd_mode::none);
        const auto head = head_sum_marks(*base);
        EXPECT_TRUE(head.count("W") != 0 && head.count("P") != 0) << "vacuity: the HEAD walk must reach P through the cbo concat";
        EXPECT_EQ(head.count("C"), 0u) << "HEAD does not mark the concat holder";
    }
    expect_defect_b_flag_diff(eng, t, cfg, {"C"}, "A6d");
    auto legacy = build_to_pmd(eng, t, cfg, pmd_mode::legacy_only);
    auto full = build_to_pmd(eng, t, cfg, pmd_mode::full);
    EXPECT_TRUE(take_snapshot(*legacy).restrictions == take_snapshot(*full).restrictions);
    EXPECT_TRUE(full->get_node("C").can_be_optimized());
    EXPECT_FALSE(full->get_node("C").can_share_buffer());
}

// (HEAD comparison via head_sum_marks: valid only for graphs without stateless_kv.)
// A14b: purity of alias_analysis::build on the plain sum graph (current bmd on both sides), plus the HEAD comparison for the same graph:
// X is both the HEAD root and the owner, so the transition adds nothing.
TEST(alias_analysis, A14b_build_is_pure_read_sum_graph_and_no_flag_added_vs_head) {
    AA_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    ov::intel_gpu::ImplForcingMap forced;
    auto t = sum_topology(eng, forced);
    auto cfg = opt_cfg(eng, forced);
    auto legacy = build_to_pmd(eng, t, cfg, pmd_mode::legacy_only);
    auto full = build_to_pmd(eng, t, cfg, pmd_mode::full);
    const auto sl = take_snapshot(*legacy);
    const auto sf = take_snapshot(*full);
    EXPECT_TRUE(sl == sf) << diff_snapshot(sl, sf);
    expect_defect_b_flag_diff(eng, t, cfg, {}, "A14b");
}

// A7. binary_mul_inplace: FC with a fused prod. The writer is a MAY_INPLACE member (the impl may still veto it, F17 G-impl).
TEST(alias_analysis, A7_mul_inplace_is_may_inplace) {
    AA_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    topology t;
    const int64_t ifm = 64, ofm = 32;
    t.add(input_layout("input", layout{ov::PartialShape{16, ifm}, data_types::f16, format::bfyx}));
    t.add(input_layout("mul_in", layout{ov::PartialShape{16, ofm}, data_types::f16, format::bfyx}));
    t.add(data("weights", make_weights(eng, ofm, ifm, 0)));
    t.add(activation("mul_data", input_info("mul_in"), activation_func::abs));
    t.add(fully_connected("fc", input_info("input"), "weights", "", 2));
    t.add(eltwise("mul", input_info("fc"), input_info("mul_data"), eltwise_mode::prod));
    t.add(softmax("softmax", input_info("mul"), 1));
    t.add(softmax("x", input_info("softmax"), 1));
    t.add(eltwise("sum", input_info("mul"), input_info("x"), eltwise_mode::sum));
    t.add(reorder("out", input_info("sum"), format::bfyx, data_types::f32));
    ov::intel_gpu::ImplForcingMap forced{{"fc", ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::onednn}}};
    auto prog = program::build_program(eng, t, opt_cfg(eng, forced));
    ASSERT_NE(prog->get_alias_analysis(), nullptr);
    const auto& aa = *prog->get_alias_analysis();
    const auto idx = onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(prog->get_node("fc"));
    ASSERT_NE(idx, -1) << "precondition: the in-place mul is taken\n" << dump_flags(*prog);
    ASSERT_EQ(prog->get_node("fc").get_dependency(idx).id(), "mul_data");
    GET_CLASS(c, aa, prog, "mul_data");
    EXPECT_EQ(c, aa.class_of(prog->get_node("fc"))) << dump_aa(aa);
    EXPECT_EQ(c->root_node, &prog->get_node("mul_data")) << dump_aa(aa);
    EXPECT_KIND(c, prog, "fc", alias_kind::MAY_INPLACE, offset_tag::ZERO, alias_role::may_allocate_member);
}

#else  // ENABLE_ONEDNN_FOR_GPU

TEST(alias_analysis, A6_A7_need_onednn) {
    GTEST_SKIP() << "built without oneDNN: the sum / mul in-place writer tests (A6a-c, A7, A14b) are not covered";
}

#endif  // ENABLE_ONEDNN_FOR_GPU
