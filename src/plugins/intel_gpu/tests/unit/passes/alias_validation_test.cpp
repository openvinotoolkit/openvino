// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Phase 0.5b: provider edge table (E*), exec_axis (X*), compile-time R/V rules and war_safe (W*, R*, V1).
// Design: /home/ceciliapeng/repos/reports/memman/layer1/13_p05b_design.zh-CN.md (r1 + Appendix B r2).
//
// Hand graphs are built with no_optimizations; flags (can_be_optimized / runtime_skippable / output / impl type) are set by the test, so
// the engine and the rules are exercised as pure functions of the node flags. Tests that need a oneDNN in-place writer are GTEST_SKIPped
// without supports_immad (a SKIP counts as "not covered").

#include "test_utils.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "alias_analysis.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "exec_axis.h"
#include "runtime_concat_order.h"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/non_max_suppression.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/kv_cache.hpp"
#include "intel_gpu/primitives/lora.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/stateless_kv.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "pass_manager.h"
#include "program_helpers.h"
#include "program_node.h"
#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

namespace {

ExecutionConfig make_cfg(engine& eng, QueueTypes q = QueueTypes::in_order, bool optimize = false, const std::vector<std::string>& outputs = {}) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(optimize));
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

program::ptr make_hand(engine& eng, const topology& t, const std::vector<std::string>& outs = {}, QueueTypes q = QueueTypes::in_order,
                       bool internal = false, bool body = false) {
    auto prog = program::build_program(eng, t, make_cfg(eng, q, false, outs), internal, /*no_optimizations=*/true, body);
    for (auto* n : prog->get_processing_order()) {
        n->get_output_layouts(false);
        n->set_unique_id();
    }
    return prog;
}

layout f32_l(const ov::PartialShape& s = ov::PartialShape{1, 4, 8, 8}) {
    return layout{s, data_types::f32, format::bfyx};
}

topology chain_topology() {
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(activation("R", input_info("P"), activation_func::abs));
    t.add(activation("Q", input_info("R"), activation_func::abs));
    t.add(activation("O", input_info("Q"), activation_func::abs));
    return t;
}

topology concat_topology() {
    topology t;
    t.add(input_layout("in1", f32_l()));
    t.add(input_layout("in2", f32_l()));
    t.add(activation("P1", input_info("in1"), activation_func::relu));
    t.add(activation("P2", input_info("in2"), activation_func::abs));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1, data_types::f32));
    t.add(activation("O", input_info("C"), activation_func::abs));
    return t;
}

topology crop_topology() {
    topology t;
    t.add(input_layout("in", f32_l(ov::PartialShape{1, 16, 8, 8})));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(crop("K", input_info("P"), tensor(1, 8, 8, 8), tensor(0, 0, 0, 0)));
    t.add(activation("O", input_info("K"), activation_func::abs));
    return t;
}

topology dyn_concat_topology() {
    topology t;
    t.add(input_layout("in1", layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}));
    t.add(input_layout("in2", layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}));
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(activation("R", input_info("C"), activation_func::abs));
    t.add(reorder("O", input_info("R"), format::bfyx, data_types::f32));
    return t;
}

topology kv_topology(bool state_dyn, const std::string& rv_variable, const std::string& kv_variable) {
    const ov::Dimension seq = state_dyn ? ov::Dimension::dynamic() : ov::Dimension(4);
    const ov::PartialShape state_shape{1, 2, seq, 8};
    topology t;
    t.add(input_layout("state", layout{state_shape, data_types::f16, format::bfyx}));
    t.add(read_value{"rv", {input_info("state")}, rv_variable, {layout{state_shape, data_types::f16, format::bfyx}}});
    t.add(input_layout("new_kv", layout{ov::PartialShape{1, 2, state_dyn ? ov::Dimension::dynamic() : ov::Dimension(1), 8}, data_types::f16, format::bfyx}));
    t.add(kv_cache("kv", {input_info("rv"), input_info("new_kv")}, ov::op::util::VariableInfo{state_shape, ov::element::f16, kv_variable}, 2, 0, false, false, false));
    t.add(activation("O", input_info("kv"), activation_func::abs));
    return t;
}

topology lora_topology() {
    topology t;
    t.add(input_layout("main", layout{ov::PartialShape{1, 1, 256}, data_types::f32, format::bfyx}));
    t.add(input_layout("input", layout{ov::PartialShape{1, 1, 128}, data_types::f32, format::bfyx}));
    t.add(input_layout("a", layout{ov::PartialShape{16, 128}, data_types::f32, format::bfyx}));
    t.add(input_layout("alpha", layout{ov::PartialShape{1, 16}, data_types::f32, format::bfyx}));
    t.add(input_layout("b", layout{ov::PartialShape{256, 16}, data_types::f32, format::bfyx}));
    t.add(activation("fc_like", input_info("main"), activation_func::relu));
    t.add(lora("lora", {input_info("fc_like"), input_info("input"), input_info("a"), input_info("alpha"), input_info("b")}, true));
    t.add(activation("O", input_info("lora"), activation_func::abs));
    return t;
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
        os << "] readers=[";
        for (const auto* r : c.readers) {
            os << r->id() << " ";
        }
        os << "]\n";
    }
    return os.str();
}

// A fixed set of hand graphs whose partition was recorded at HEAD (911e505b01), before the provider edge table existed.
std::string e2_dump_all() {
    auto& eng = get_test_engine();
    std::ostringstream os;
    auto add = [&](const std::string& name, program::ptr prog) {
        os << "### " << name << "\n" << dump_aa(alias_analysis::build(*prog));
    };
    {
        auto p = make_hand(eng, chain_topology(), {"O"});
        p->get_node("R").can_be_optimized(true);
        p->get_node("Q").can_be_optimized(true);
        add("chain", p);
    }
    {
        auto p = make_hand(eng, concat_topology(), {"O"});
        p->get_node("C").can_be_optimized(true);
        add("concat", p);
    }
    {
        auto p = make_hand(eng, crop_topology(), {"O"});
        p->get_node("K").can_be_optimized(true);
        add("crop", p);
    }
    {
        auto p = make_hand(eng, dyn_concat_topology(), {"O"});
        p->get_node("C").can_be_optimized(true);
        p->get_node("C").set_runtime_skippable(true);
        p->get_node("R").set_runtime_skippable(true);
        add("dyn_concat", p);
    }
    for (const bool cbo : {true, false}) {
        auto p = make_hand(eng, kv_topology(false, "v0", "v0"), {"O"});
        p->get_node("kv").can_be_optimized(cbo);
        add(cbo ? "kv_cbo" : "kv_non_cbo", p);
    }
    add("kv_dynamic", make_hand(eng, kv_topology(true, "v0", "v0"), {"O"}));
    add("kv_other_variable", make_hand(eng, kv_topology(false, "other", "v0"), {"O"}));
    add("lora", make_hand(eng, lora_topology(), {"O"}));
    add("chain_oooq", make_hand(eng, chain_topology(), {"O"}, QueueTypes::out_of_order));
    return os.str();
}

}  // namespace

// ==============================================================================================================================
// E2. build() without extra edges keeps the class partition of HEAD 911e505b01 (golden recorded before the provider edge table).
// ==============================================================================================================================
TEST(alias_validation, E2_partition_matches_head_golden) {
    static const char* const golden = R"GOLD(### chain
class root=in:0 fb=0x0 dm=0x1 place=1 members=[in:0/k0/o0/r0 ] readers=[P ]
class root=P:0 fb=0x0 dm=0x0 place=4 members=[P:0/k0/o0/r0 R:0/k1/o0/r1 Q:0/k1/o0/r1 ] readers=[O ]
class root=O:0 fb=0x0 dm=0x4 place=3 members=[O:0/k0/o0/r0 ] readers=[]
### concat
class root=in2:0 fb=0x0 dm=0x1 place=1 members=[in2:0/k0/o0/r0 ] readers=[P2 ]
class root=C:0 fb=0x0 dm=0x0 place=4 members=[P2:0/k3/o1/r1 P1:0/k3/o1/r1 C:0/k0/o0/r0 ] readers=[O ]
class root=in1:0 fb=0x0 dm=0x1 place=1 members=[in1:0/k0/o0/r0 ] readers=[P1 ]
class root=O:0 fb=0x0 dm=0x4 place=3 members=[O:0/k0/o0/r0 ] readers=[]
### crop
class root=in:0 fb=0x0 dm=0x1 place=1 members=[in:0/k0/o0/r0 ] readers=[P ]
class root=P:0 fb=0x0 dm=0x0 place=4 members=[P:0/k0/o0/r0 K:0/k1/o2/r1 ] readers=[O ]
class root=O:0 fb=0x0 dm=0x4 place=3 members=[O:0/k0/o0/r0 ] readers=[]
### dyn_concat
class root=in2:0 fb=0x0 dm=0x1 place=1 members=[in2:0/k0/o0/r0 ] readers=[P2 ]
class root=C:0 fb=0x0 dm=0x0 place=4 members=[P2:0/k5/o3/r2 P1:0/k5/o3/r2 C:0/k0/o0/r0 R:0/k4/o0/r2 ] readers=[O ]
class root=in1:0 fb=0x0 dm=0x1 place=1 members=[in1:0/k0/o0/r0 ] readers=[P1 ]
class root=O:0 fb=0x0 dm=0x4 place=3 members=[O:0/k0/o0/r0 ] readers=[]
### kv_cbo
class root=state:0 fb=0x0 dm=0x1 place=1 members=[state:0/k0/o0/r0 ] readers=[rv ]
class root=rv:0 fb=0x40 dm=0x2 place=0 members=[rv:0/k0/o0/r0 kv:0/k1/o0/r1 ] readers=[O ]
class root=new_kv:0 fb=0x0 dm=0x1 place=1 members=[new_kv:0/k0/o0/r0 ] readers=[kv ]
class root=O:0 fb=0x0 dm=0x4 place=3 members=[O:0/k0/o0/r0 ] readers=[]
### kv_non_cbo
class root=state:0 fb=0x0 dm=0x1 place=1 members=[state:0/k0/o0/r0 ] readers=[rv ]
class root=rv:0 fb=0x40 dm=0x2 place=0 members=[rv:0/k0/o0/r0 kv:0/k6/o0/r2 ] readers=[O ]
class root=new_kv:0 fb=0x0 dm=0x1 place=1 members=[new_kv:0/k0/o0/r0 ] readers=[kv ]
class root=O:0 fb=0x0 dm=0x4 place=3 members=[O:0/k0/o0/r0 ] readers=[]
### kv_dynamic
class root=state:0 fb=0x4 dm=0x1 place=0 members=[state:0/k0/o0/r0 ] readers=[rv ]
class root=rv:0 fb=0x4 dm=0x2 place=0 members=[rv:0/k0/o0/r0 kv:0/k6/o0/r2 ] readers=[O ]
class root=new_kv:0 fb=0x0 dm=0x1 place=1 members=[new_kv:0/k0/o0/r0 ] readers=[kv ]
class root=O:0 fb=0x0 dm=0x4 place=3 members=[O:0/k0/o0/r0 ] readers=[]
### kv_other_variable
class root=state:0 fb=0x0 dm=0x1 place=1 members=[state:0/k0/o0/r0 ] readers=[rv ]
class root=rv:0 fb=0x0 dm=0x2 place=2 members=[rv:0/k0/o0/r0 ] readers=[kv ]
class root=new_kv:0 fb=0x0 dm=0x1 place=1 members=[new_kv:0/k0/o0/r0 ] readers=[kv ]
class root=kv:0 fb=0x40 dm=0x2 place=0 members=[kv:0/k0/o0/r0 ] readers=[O ]
class root=O:0 fb=0x0 dm=0x4 place=3 members=[O:0/k0/o0/r0 ] readers=[]
### lora
class root=main:0 fb=0x0 dm=0x1 place=1 members=[main:0/k0/o0/r0 ] readers=[fc_like ]
class root=fc_like:0 fb=0x40 dm=0x0 place=0 members=[fc_like:0/k0/o0/r0 lora:0/k6/o0/r2 ] readers=[O ]
class root=input:0 fb=0x0 dm=0x1 place=1 members=[input:0/k0/o0/r0 ] readers=[lora ]
class root=b:0 fb=0x0 dm=0x1 place=1 members=[b:0/k0/o0/r0 ] readers=[lora ]
class root=alpha:0 fb=0x0 dm=0x1 place=1 members=[alpha:0/k0/o0/r0 ] readers=[lora ]
class root=a:0 fb=0x0 dm=0x1 place=1 members=[a:0/k0/o0/r0 ] readers=[lora ]
class root=O:0 fb=0x0 dm=0x4 place=3 members=[O:0/k0/o0/r0 ] readers=[]
### chain_oooq
class root=in:0 fb=0x1 dm=0x1 place=0 members=[in:0/k0/o0/r0 ] readers=[P ]
class root=P:0 fb=0x1 dm=0x0 place=0 members=[P:0/k0/o0/r0 ] readers=[R ]
class root=R:0 fb=0x1 dm=0x0 place=0 members=[R:0/k0/o0/r0 ] readers=[Q ]
class root=Q:0 fb=0x1 dm=0x0 place=0 members=[Q:0/k0/o0/r0 ] readers=[O ]
class root=O:0 fb=0x1 dm=0x4 place=0 members=[O:0/k0/o0/r0 ] readers=[]
)GOLD";
    EXPECT_EQ(e2_dump_all(), std::string(golden));
    if (const char* path = std::getenv("ALIAS_E2_PRINT")) {
        std::ofstream(path) << e2_dump_all();
    }
}

// ==============================================================================================================================
// E1. provider edge table and member fields.
// ==============================================================================================================================
namespace {
const alias_edge* the_edge_from(const alias_analysis& aa, const program_node& member, uint16_t port, size_t nth = 0) {
    const auto& v = aa.edges_from(member, port);
    return nth < v.size() ? &aa.edges()[v[nth]] : nullptr;
}
const alias_member* member_of(const alias_analysis& aa, const program_node& n, uint16_t port = 0) {
    const auto* c = aa.class_of(n, port);
    if (!c) {
        return nullptr;
    }
    for (const auto& m : c->members) {
        if (m.node == &n && m.id.index == port) {
            return &m;
        }
    }
    return nullptr;
}
}  // namespace

TEST(alias_validation, E1_view_chain_edges) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, chain_topology(), {"O"});
    prog->get_node("R").can_be_optimized(true);
    prog->get_node("Q").can_be_optimized(true);
    auto aa = alias_analysis::build(*prog);
    ASSERT_EQ(aa.edges().size(), 2u);
    const auto* e = the_edge_from(aa, prog->get_node("R"), 0);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->provider_node, &prog->get_node("P"));
    EXPECT_EQ(e->provider_port, 0);
    EXPECT_EQ(e->kind, alias_kind::MUST_VIEW);
    EXPECT_EQ(e->off, offset_tag::ZERO);
    EXPECT_EQ(e->act, edge_activity::ALWAYS);
    EXPECT_EQ(e->dep_index, 0);
    EXPECT_EQ(e->origin, edge_origin::view);
    EXPECT_EQ(aa.edges_into(prog->get_node("P"), 0).size(), 1u);
    EXPECT_EQ(aa.edges_into(prog->get_node("R"), 0).size(), 1u);
    EXPECT_EQ(aa.edges_into(prog->get_node("Q"), 0).size(), 0u);
    EXPECT_TRUE(aa.edges_from(prog->get_node("P"), 0).empty());
    EXPECT_TRUE(aa.edges_from(prog->get_node("P"), 7).empty()) << "unknown port";
    const auto* mr = member_of(aa, prog->get_node("R"));
    ASSERT_NE(mr, nullptr);
    EXPECT_EQ(mr->edge_count, 1);
    EXPECT_GE(mr->first_edge, 0);
    EXPECT_EQ(mr->provider_node, &prog->get_node("P"));
    EXPECT_FALSE(mr->opaque);
    const auto* mp = member_of(aa, prog->get_node("P"));
    ASSERT_NE(mp, nullptr);
    EXPECT_EQ(mp->edge_count, 0);
    EXPECT_EQ(mp->first_edge, -1);
    EXPECT_EQ(mp->provider_node, nullptr);
}

TEST(alias_validation, E1_concat_and_crop_edges) {
    auto& eng = get_test_engine();
    {
        auto prog = make_hand(eng, concat_topology(), {"O"});
        prog->get_node("C").can_be_optimized(true);
        auto aa = alias_analysis::build(*prog);
        ASSERT_EQ(aa.edges().size(), 2u);
        for (const char* id : {"P1", "P2"}) {
            const auto* e = the_edge_from(aa, prog->get_node(id), 0);
            ASSERT_NE(e, nullptr) << id;
            EXPECT_EQ(e->provider_node, &prog->get_node("C"));
            EXPECT_EQ(e->kind, alias_kind::MUST_SUBREGION);
            EXPECT_EQ(e->off, offset_tag::LAYOUT_PADDING);
            EXPECT_EQ(e->origin, edge_origin::concat_slot);
            EXPECT_EQ(e->dep_index, -1);
        }
        EXPECT_EQ(aa.edges_into(prog->get_node("C"), 0).size(), 2u);
    }
    {
        auto prog = make_hand(eng, crop_topology(), {"O"});
        prog->get_node("K").can_be_optimized(true);
        auto aa = alias_analysis::build(*prog);
        const auto* e = the_edge_from(aa, prog->get_node("K"), 0);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->off, offset_tag::STATIC_BYTES);
        EXPECT_EQ(e->origin, edge_origin::view);
        EXPECT_EQ(e->provider_node, &prog->get_node("P"));
    }
    {   // dynamic: MAYBE edges, RUNTIME offsets
        auto prog = make_hand(eng, dyn_concat_topology(), {"O"});
        prog->get_node("C").can_be_optimized(true);
        prog->get_node("C").set_runtime_skippable(true);
        prog->get_node("R").set_runtime_skippable(true);
        auto aa = alias_analysis::build(*prog);
        const auto* e = the_edge_from(aa, prog->get_node("P1"), 0);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->kind, alias_kind::MAY_SUBREGION);
        EXPECT_EQ(e->off, offset_tag::RUNTIME);
        EXPECT_EQ(e->act, edge_activity::MAYBE);
        const auto* v = the_edge_from(aa, prog->get_node("R"), 0);
        ASSERT_NE(v, nullptr);
        EXPECT_EQ(v->kind, alias_kind::MAY_VIEW);
        EXPECT_EQ(v->act, edge_activity::MAYBE);
    }
}

TEST(alias_validation, E1_kv_lora_stateless_assign) {
    auto& eng = get_test_engine();
    {   // kv_cache, non cbo: MAY_INPLACE edge to the read_value, registered persistent writer
        auto prog = make_hand(eng, kv_topology(false, "v0", "v0"), {"O"});
        auto aa = alias_analysis::build(*prog);
        const auto* e = the_edge_from(aa, prog->get_node("kv"), 0);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->origin, edge_origin::kv_cache);
        EXPECT_EQ(e->kind, alias_kind::MAY_INPLACE);
        EXPECT_EQ(e->provider_node, &prog->get_node("rv"));
        EXPECT_EQ(e->dep_index, 0);
        const auto* b = aa.binding_written_by(prog->get_node("kv"));
        ASSERT_NE(b, nullptr);
        EXPECT_EQ(b->key.k, storage_key::kind::variable);
        EXPECT_EQ(b->key.variable_id, "v0");
        ASSERT_EQ(b->aliases.size(), 1u);
        EXPECT_EQ(b->aliases[0].first, &prog->get_node("rv"));
        ASSERT_EQ(b->writers.size(), 1u);
        EXPECT_EQ(b->writers[0].kind, persistent_writer_kind::kv_cache_inplace);
    }
    {   // kv_cache, cbo: the settled view edge already joined the classes, the writer is registered anyway
        auto prog = make_hand(eng, kv_topology(false, "v0", "v0"), {"O"});
        prog->get_node("kv").can_be_optimized(true);
        auto aa = alias_analysis::build(*prog);
        const auto* e = the_edge_from(aa, prog->get_node("kv"), 0);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->origin, edge_origin::view);
        EXPECT_EQ(aa.edges_from(prog->get_node("kv"), 0).size(), 1u);
        ASSERT_NE(aa.binding_written_by(prog->get_node("kv")), nullptr) << "registered although no kv edge was added";
    }
    {   // dep 0 of another variable: no binding match, opaque
        auto prog = make_hand(eng, kv_topology(false, "other", "v0"), {"O"});
        auto aa = alias_analysis::build(*prog);
        EXPECT_TRUE(aa.tensor_is_opaque(prog->get_node("kv"), 0));
        EXPECT_EQ(aa.binding_written_by(prog->get_node("kv")), nullptr);
        EXPECT_NE(aa.class_of(prog->get_node("kv"))->role_conflict_causes & rc_opaque, 0u);
    }
    {   // N3: compressed kv_cache: record what the engine sees (port count) and the state port writers
        auto t = kv_topology(false, "v0", "v0");
        auto prog = make_hand(eng, t, {"O"});
        const_cast<kv_cache&>(*std::static_pointer_cast<const kv_cache>(prog->get_node("kv").get_primitive())).compressed = true;
        auto aa = alias_analysis::build(*prog);
        const size_t outs = prog->get_node("kv").get_outputs_count();
        std::cout << "[N3] compressed kv_cache get_outputs_count()=" << outs << std::endl;
        const auto* b = aa.binding_written_by(prog->get_node("kv"));
        ASSERT_NE(b, nullptr);
        size_t state_ports = 0;
        for (const auto& w : b->writers) {
            state_ports += w.kind == persistent_writer_kind::kv_cache_state_port ? 1 : 0;
        }
        EXPECT_EQ(state_ports, outs > 2 ? outs - 2 : 0u);
        EXPECT_TRUE(aa.tensor_is_opaque(prog->get_node("kv"), 0));
    }
    {   // lora
        auto prog = make_hand(eng, lora_topology(), {"O"});
        auto aa = alias_analysis::build(*prog);
        const auto* e = the_edge_from(aa, prog->get_node("lora"), 0);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->origin, edge_origin::lora);
        EXPECT_EQ(e->kind, alias_kind::MAY_INPLACE);
        EXPECT_TRUE(aa.tensor_is_opaque(prog->get_node("lora"), 0));
        EXPECT_TRUE(member_of(aa, prog->get_node("lora"))->opaque);
        EXPECT_NE(aa.class_of(prog->get_node("lora"))->role_conflict_causes & rc_opaque, 0u);
    }
    {   // assign: copy writer of the variable, read_value is the alias
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(read_value{"rv", {input_info("in")}, "v0", {f32_l()}});
        t.add(activation("X", input_info("rv"), activation_func::abs));
        t.add(assign("as", {input_info("X")}, "v0", f32_l()));
        auto prog = make_hand(eng, t, {"as"});
        auto aa = alias_analysis::build(*prog);
        const auto* b = aa.binding_written_by(prog->get_node("as"));
        ASSERT_NE(b, nullptr);
        EXPECT_EQ(b->writers[0].kind, persistent_writer_kind::assign_copy);
        ASSERT_EQ(b->aliases.size(), 1u);
        EXPECT_EQ(b->aliases[0].first, &prog->get_node("rv"));
        EXPECT_TRUE(aa.edges_from(prog->get_node("as"), 0).empty()) << "assign is a copy, not an alias";
    }
    {   // stateless_kv: persistent writer keyed by the external node owning `past`
        topology t;
        t.add(input_layout("past", layout{ov::PartialShape{1, 2, -1, 4}, data_types::f32, format::bfyx}));
        t.add(input_layout("new_token", layout{ov::PartialShape{1, 2, 1, 4}, data_types::f32, format::bfyx}));
        t.add(input_layout("present_len", layout{ov::PartialShape{1}, data_types::i64, format::bfyx}));
        auto kv = stateless_kv("stateless_kv", {input_info("past"), input_info("new_token"), input_info("present_len")}, 2, true);
        kv.num_outputs = 2;
        kv.output_data_types = {data_types::f32, data_types::f32};
        t.add(kv);
        t.add(reorder("result", input_info("stateless_kv", 0), format::bfyx, data_types::f32));
        auto prog = make_hand(eng, t, {"result"});
        auto aa = alias_analysis::build(*prog);
        const auto* e = the_edge_from(aa, prog->get_node("stateless_kv"), 0);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->origin, edge_origin::stateless_kv);
        const auto* b = aa.binding_written_by(prog->get_node("stateless_kv"));
        ASSERT_NE(b, nullptr);
        EXPECT_EQ(b->key.k, storage_key::kind::external_node);
        EXPECT_EQ(b->key.node, &prog->get_node("past"));
        EXPECT_EQ(b->writers[0].kind, persistent_writer_kind::stateless_kv_past);
        EXPECT_TRUE(aa.tensor_is_opaque(prog->get_node("stateless_kv"), 1)) << "multi-output stateless_kv is opaque on all ports";
    }
}

TEST(alias_validation, E1_nms_gather_port_k_edges) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("boxes", layout{ov::PartialShape{1, 8, 4}, data_types::f32, format::bfyx}));
    t.add(input_layout("scores", layout{ov::PartialShape{1, 2, 8}, data_types::f32, format::bfyx}));
    t.add(cldnn::data("num_per_class", eng.allocate_memory({data_types::i32, format::bfyx, {1, 1, 1, 1}})));
    t.add(cldnn::data("iou_threshold", eng.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}})));
    t.add(cldnn::data("score_threshold", eng.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}})));
    auto nms = non_max_suppression("nms", input_info("boxes"), input_info("scores"), 16, false, true, "num_per_class", "iou_threshold",
                                   "score_threshold", "", "", "", 3);
    nms.output_data_types = {optional_data_type{}, optional_data_type{data_types::f32}, optional_data_type{}};
    nms.output_paddings = {padding(), padding(), padding()};
    t.add(nms);
    t.add(non_max_suppression_gather("nms_gather", {input_info("nms", 0), input_info("nms", 1), input_info("nms", 2)}, 3));
    t.add(reorder("out0", input_info("nms_gather", 0), format::bfyx, data_types::i32));
    t.add(reorder("out1", input_info("nms_gather", 1), format::bfyx, data_types::f32));
    t.add(reorder("out2", input_info("nms_gather", 2), format::bfyx, data_types::i32));
    auto prog = make_hand(eng, t, {"out0", "out1", "out2"});
    auto aa = alias_analysis::build(*prog);
    for (uint16_t k = 0; k < 3; ++k) {
        const auto* e = the_edge_from(aa, prog->get_node("nms_gather"), k);
        ASSERT_NE(e, nullptr) << "port " << k;
        EXPECT_EQ(e->provider_node, &prog->get_node("nms"));
        EXPECT_EQ(e->provider_port, k);
        EXPECT_EQ(e->dep_index, k);
        EXPECT_EQ(e->origin, edge_origin::view);
    }
}

// ---- hypothetical edges (N13) and role conflict causes ----
TEST(alias_validation, E1_hypothetical_inplace_edges_and_n13) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, chain_topology(), {"O"});
    const auto base = alias_analysis::build(*prog);
    EXPECT_NE(base.class_of(prog->get_node("P")), base.class_of(prog->get_node("O")));
    std::vector<hypothetical_inplace> extra{{&prog->get_node("O"), 0}};
    auto with = alias_analysis::build(*prog, compile_time_envelope{}, &extra);
    EXPECT_EQ(with.class_of(prog->get_node("O")), with.class_of(prog->get_node("Q")));
    const auto* e = the_edge_from(with, prog->get_node("O"), 0);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->origin, edge_origin::hypothetical);
    EXPECT_EQ(e->kind, alias_kind::MUST_INPLACE);
    EXPECT_EQ(e->dep_index, 0);
    EXPECT_TRUE(with.rejected_hypotheticals().empty());
    EXPECT_TRUE(base.edges().empty());
    // N13: layout mismatch (concat output is twice the size of its predecessor) -> not added, recorded
    auto cp = make_hand(eng, concat_topology(), {"O"});
    std::vector<hypothetical_inplace> bad{{&cp->get_node("C"), 0}, {&cp->get_node("C"), 5}, {nullptr, 0}};
    auto rej = alias_analysis::build(*cp, compile_time_envelope{}, &bad);
    EXPECT_EQ(rej.rejected_hypotheticals().size(), 3u);
    EXPECT_TRUE(rej.edges().empty());
    EXPECT_NE(rej.class_of(cp->get_node("C")), rej.class_of(cp->get_node("P1")));
}

TEST(alias_validation, E1_for_each_tensor_user_enumerates_each_dependency_index) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(eltwise("E", {input_info("P"), input_info("P")}, eltwise_mode::sum));
    t.add(activation("O", input_info("P"), activation_func::abs));
    auto prog = make_hand(eng, t, {"E", "O"});
    auto aa = alias_analysis::build(*prog);
    std::vector<std::pair<std::string, size_t>> seen;
    aa.for_each_tensor_user(prog->get_node("P"), 0, [&](const program_node& u, size_t d) { seen.emplace_back(u.id(), d); });
    std::sort(seen.begin(), seen.end());
    ASSERT_EQ(seen.size(), 3u);
    EXPECT_EQ(seen[0], (std::pair<std::string, size_t>{"E", 0}));
    EXPECT_EQ(seen[1], (std::pair<std::string, size_t>{"E", 1}));
    EXPECT_EQ(seen[2], (std::pair<std::string, size_t>{"O", 0}));
    size_t n1 = 0;
    aa.for_each_tensor_user(prog->get_node("P"), 1, [&](const program_node&, size_t) { ++n1; });
    EXPECT_EQ(n1, 0u);
}

TEST(alias_validation, E1_role_conflict_causes) {
    auto& eng = get_test_engine();
    {   // two allocators in one class (kv_cache as concat slot AND variable alias, A18g)
        topology t = kv_topology(false, "v0", "v0");
        t.add(input_layout("other", layout{ov::PartialShape{1, 2, 5, 8}, data_types::f16, format::bfyx}));
        t.add(concatenation("C", {input_info("kv"), input_info("other")}, 2));
        t.add(activation("O2", input_info("C"), activation_func::abs));
        auto prog = make_hand(eng, t, {"O", "O2"});
        prog->get_node("C").can_be_optimized(true);
        auto aa = alias_analysis::build(*prog);
        const auto* c = aa.class_of(prog->get_node("kv"));
        ASSERT_NE(c, nullptr);
        EXPECT_NE(c->role_conflict_causes & rc_allocators_not_one, 0u);
        const auto* ms = member_of(aa, prog->get_node("kv"));
        ASSERT_NE(ms, nullptr);
        EXPECT_EQ(ms->edge_count, 2) << "A18g: concat slot edge plus kv alias edge";
    }
    {   // a plain class has none
        auto prog = make_hand(eng, chain_topology(), {"O"});
        prog->get_node("R").can_be_optimized(true);
        auto aa = alias_analysis::build(*prog);
        for (const auto& c : aa.classes()) {
            EXPECT_EQ(c.role_conflict_causes, 0u);
        }
    }
}

// ==============================================================================================================================
// X1-X4. exec_axis
// ==============================================================================================================================
namespace {
topology branch_topology() {
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

program::ptr build_to_post(engine& eng, const topology& t, const ExecutionConfig& cfg, bool run_post = true) {
    auto prog = program::build_program(eng, t, cfg, /*is_internal=*/false, /*no_optimizations=*/true);
    program_wrapper::pre_optimize_graph(*prog, false);
    program_wrapper::run_graph_compilation(*prog);
    if (run_post) {
        program_wrapper::post_optimize_graph(*prog, false);
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
}  // namespace

TEST(alias_validation, X1_static_network_axis_equals_exec_order) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(cldnn::data("w", eng.allocate_memory({data_types::f32, format::bfyx, {1, 4, 8, 8}})));
    t.add(eltwise("E", {input_info("in"), input_info("w")}, eltwise_mode::sum));
    t.add(activation("O", input_info("E"), activation_func::abs));
    auto cfg = make_cfg(eng, QueueTypes::in_order, true, {"O"});
    auto prog = program::build_program(eng, t, cfg, false, false);
    network net(prog, 0);
    const auto ax = exec_axis::build(*prog);
    ASSERT_TRUE(ax.valid());
    const auto ids = net.get_executed_primitive_ids();
    EXPECT_EQ(ax.size(), ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        ASSERT_TRUE(ax.position(prog->get_node(ids[i])).has_value()) << ids[i];
        EXPECT_EQ(*ax.position(prog->get_node(ids[i])), i) << ids[i];
    }
    // data nodes are not executable
    EXPECT_FALSE(ax.position(prog->get_node("w")).has_value());
}

TEST(alias_validation, X2_dynamic_runtime_concat_axis_equals_exec_order) {
    auto& eng = get_test_engine();
    auto prog = build_to_post(eng, branch_topology(), make_cfg(eng, QueueTypes::in_order, true));
    ASSERT_TRUE(prog->get_node("C").can_be_optimized());
    ASSERT_TRUE(prog->get_node("C").is_dynamic());
    program_wrapper::prepare_memory_dependencies(*prog);
    network net(prog, 0);
    const auto ids = net.get_executed_primitive_ids();
    const auto ax = exec_axis::build(*prog);
    ASSERT_TRUE(ax.valid());
    ASSERT_EQ(ax.size(), ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        EXPECT_EQ(ax.position(prog->get_node(ids[i])), std::optional<uint32_t>(static_cast<uint32_t>(i))) << ids[i];
    }
    // the predecessors of the runtime concat sit directly in front of it, in dependency order
    EXPECT_EQ(*ax.position(prog->get_node("C")), *ax.position(prog->get_node("P1")) + 1);
    EXPECT_EQ(*ax.position(prog->get_node("P1")), *ax.position(prog->get_node("P0")) + 1);
}

TEST(alias_validation, X3_out_of_order_queue_has_no_axis) {
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng, QueueTypes::out_of_order, false, {"O"});
    auto prog = make_hand(eng, chain_topology(), {"O"}, QueueTypes::out_of_order);
    ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::out_of_order);
    const auto ax = exec_axis::build(*prog);
    EXPECT_EQ(ax.get_status(), exec_axis::status::not_in_order);
    EXPECT_FALSE(ax.valid());
    EXPECT_FALSE(ax.position(prog->get_node("P")).has_value());
}

TEST(alias_validation, X4_broken_normalization_is_reported) {
    auto& eng = get_test_engine();
    auto prog = build_to_post(eng, branch_topology(), make_cfg(eng, QueueTypes::in_order, true), /*run_post=*/false);
    ASSERT_TRUE(prog->get_node("C").can_be_optimized());
    force_order(*prog, {"in1", "D", "P0", "in2", "Z", "P1", "C"});
    ASSERT_FALSE(runtime_concat_order::is_normalized(runtime_concat_order::filtered_processing_order(prog->get_processing_order())));
    const auto ax = exec_axis::build(*prog);
    EXPECT_EQ(ax.get_status(), exec_axis::status::not_normalized);
    EXPECT_FALSE(ax.position(prog->get_node("C")).has_value());
    // for_test positions round trip
    const auto fake = exec_axis::for_test(exec_axis::status::ok, {{&prog->get_node("C"), 41}});
    EXPECT_EQ(fake.position(prog->get_node("C")), std::optional<uint32_t>(41u));
    EXPECT_FALSE(fake.position(prog->get_node("D")).has_value());
}
