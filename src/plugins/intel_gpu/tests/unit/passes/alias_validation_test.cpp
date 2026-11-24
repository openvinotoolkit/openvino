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
#include "alias_validation.h"
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
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
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
    {   // compressed kv_cache: all ports are opaque. NOT COVERED / NOT VERIFIED: whether get_outputs_count() of a real compressed kv_cache is 4
        // (the hand graph reports 1 because num_outputs is not set by the primitive); the state port writers (ports 2, 3) are therefore not
        // asserted here, only the opaque marking and the registration of the in-place writer.
        auto prog = make_hand(eng, kv_topology(false, "v0", "v0"), {"O"});
        const_cast<kv_cache&>(*std::static_pointer_cast<const kv_cache>(prog->get_node("kv").get_primitive())).compressed = true;
        auto aa = alias_analysis::build(*prog);
        ASSERT_NE(aa.binding_written_by(prog->get_node("kv")), nullptr);
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

// ==============================================================================================================================
// W*. war_safe
// ==============================================================================================================================
namespace {

struct war_run {
    alias_analysis aa;
    exec_axis ax;
    war_result res;
};

// Builds the engine (with the hypothetical in-place edges `hyp`: {writer id, dep index}) and queries the writer `id`.
war_run war_of(program& prog, const std::string& id, int dep, const std::vector<std::pair<std::string, int>>& hyp, const exec_axis* axis = nullptr) {
    std::vector<hypothetical_inplace> extra;
    for (const auto& h : hyp) {
        extra.push_back({&prog.get_node(h.first), h.second});
    }
    war_run r{alias_analysis::build(prog, compile_time_envelope{}, &extra), axis ? *axis : exec_axis::build(prog), {}};
    EXPECT_TRUE(r.aa.rejected_hypotheticals().empty()) << "a hypothetical edge was rejected (N13)";
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

#define EXPECT_VERDICT(r, v) EXPECT_EQ((r).res.verdict, v) << why((r).res)
#define EXPECT_REASON(r, bit) EXPECT_NE((r).res.reasons & (bit), 0u) << #bit << " expected; " << why((r).res)
#define EXPECT_NO_REASON(r, bit) EXPECT_EQ((r).res.reasons & (bit), 0u) << #bit << " unexpected; " << why((r).res)

// in -> P -> {W, Y, Z}; W -> O, Y -> OY, Z -> OZ
topology fan_topology(bool with_y = true, bool with_z = false) {
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(activation("W", input_info("P"), activation_func::abs));
    if (with_y) {
        t.add(activation("Y", input_info("P"), activation_func::abs));
        t.add(activation("OY", input_info("Y"), activation_func::abs));
    }
    if (with_z) {
        t.add(activation("Z", input_info("P"), activation_func::abs));
        t.add(activation("OZ", input_info("Z"), activation_func::abs));
    }
    return t;
}
std::vector<std::string> fan_outs(bool with_y = true, bool with_z = false) {
    std::vector<std::string> o{"W"};
    if (with_y) {
        o.push_back("OY");
    }
    if (with_z) {
        o.push_back("OZ");
    }
    return o;
}

}  // namespace

// W1/W2: simple residual, with and without a second reader before / after the writer.
TEST(alias_validation, W1_single_reader_is_safe) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, fan_topology(false), fan_outs(false));
    auto r = war_of(*prog, "W", 0, {{"W", 0}});
    EXPECT_VERDICT(r, war_verdict::Safe);
    EXPECT_REASON(r, wr_no_other_reader);
    EXPECT_EQ(r.res.writer_pos, static_cast<int64_t>(*r.ax.position(prog->get_node("W"))));
}

TEST(alias_validation, W2_late_reader_is_unsafe_and_early_reader_is_safe) {
    auto& eng = get_test_engine();
    {
        auto prog = make_hand(eng, fan_topology(), fan_outs());
        force_order(*prog, {"P", "W", "Y"});
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_late_reader);
        ASSERT_FALSE(r.res.witnesses.empty());
        EXPECT_EQ(r.res.witnesses[0].reader, &prog->get_node("Y"));
        EXPECT_EQ(r.res.witnesses[0].via_node, &prog->get_node("P"));
    }
    {
        auto prog = make_hand(eng, fan_topology(), fan_outs());
        force_order(*prog, {"P", "Y", "W"});
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Safe);
        EXPECT_NO_REASON(r, wr_no_other_reader);
    }
}

// W3 (F29 construction at engine level): the overwritten tensor A has a view R (cbo) whose reader Y runs after W.
TEST(alias_validation, W3_late_reader_of_a_view_of_the_overwritten_tensor) {
    auto& eng = get_test_engine();
    auto make = [&](bool late) {
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(activation("A", input_info("in"), activation_func::relu));
        t.add(activation("R", input_info("A"), activation_func::abs));
        t.add(activation("W", input_info("A"), activation_func::abs));
        t.add(activation("Y", input_info("R"), activation_func::abs));
        auto prog = make_hand(eng, t, {"W", "Y"});
        prog->get_node("R").can_be_optimized(true);
        force_order(*prog, late ? std::vector<std::string>{"A", "W", "R", "Y"} : std::vector<std::string>{"A", "R", "Y", "W"});
        return prog;
    };
    {
        auto prog = make(true);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_late_reader);
        // R itself reads A after the writer (conservative, P-1), Y reads R: it is found through the view edge going down
        bool found = false;
        for (const auto& w : r.res.witnesses) {
            found = found || (w.reader == &prog->get_node("Y") && w.via_node == &prog->get_node("R"));
        }
        EXPECT_TRUE(found) << why(r.res);
    }
    {
        auto prog = make(false);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Safe);
    }
}

// W4 (F4 row 11 at engine level): the writer reads a VIEW V; the owner O of V has another reader Z.
TEST(alias_validation, W4_late_reader_of_the_owner_of_the_overwritten_view) {
    auto& eng = get_test_engine();
    auto make = [&](bool late) {
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(activation("O1", input_info("in"), activation_func::relu));
        t.add(activation("V", input_info("O1"), activation_func::abs));
        t.add(activation("W", input_info("V"), activation_func::abs));
        t.add(activation("Z", input_info("O1"), activation_func::abs));
        auto prog = make_hand(eng, t, {"W", "Z"});
        prog->get_node("V").can_be_optimized(true);
        force_order(*prog, late ? std::vector<std::string>{"O1", "V", "W", "Z"} : std::vector<std::string>{"O1", "V", "Z", "W"});
        return prog;
    };
    {
        auto prog = make(true);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_late_reader);
        ASSERT_FALSE(r.res.witnesses.empty());
        EXPECT_EQ(r.res.witnesses[0].reader, &prog->get_node("Z"));
        EXPECT_EQ(r.res.witnesses[0].via_node, &prog->get_node("O1")) << "found through the view edge going up";
    }
    {
        auto prog = make(false);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Safe);
    }
}

// slot / holder cases (F10). C = concat(S1, S2) is an in-place holder.
namespace {
topology slots_topology() {
    topology t;
    t.add(input_layout("in1", f32_l()));
    t.add(input_layout("in2", f32_l()));
    t.add(activation("S1", input_info("in1"), activation_func::relu));
    t.add(activation("S2", input_info("in2"), activation_func::abs));
    t.add(concatenation("C", {input_info("S1"), input_info("S2")}, 1, data_types::f32));
    t.add(activation("O", input_info("C"), activation_func::abs));
    return t;
}
}  // namespace

// W5: the writer overwrites the holder: every reader of any slot after the writer is a victim; static siblings included.
TEST(alias_validation, W5_holder_overwritten_slot_reader_decides) {
    auto& eng = get_test_engine();
    auto make = [&](bool late) {
        topology t = slots_topology();
        t.add(activation("W", input_info("C"), activation_func::abs));
        t.add(activation("Z", input_info("S1"), activation_func::abs));
        auto prog = make_hand(eng, t, {"O", "W", "Z"});
        prog->get_node("C").can_be_optimized(true);
        force_order(*prog, late ? std::vector<std::string>{"S1", "S2", "C", "O", "W", "Z"} : std::vector<std::string>{"S1", "S2", "C", "O", "Z", "W"});
        return prog;
    };
    {
        auto prog = make(true);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_late_reader);
        EXPECT_EQ(r.res.witnesses.at(0).reader, &prog->get_node("Z"));
    }
    {
        auto prog = make(false);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Safe) << "all readers of the holder and of its slots run before the writer";
    }
}

// W6: overwriting one slot: the holder (and its readers) are affected, the static sibling slots are not (pruned).
TEST(alias_validation, W6_slot_overwritten_static_sibling_is_pruned_holder_is_not) {
    auto& eng = get_test_engine();
    auto make = [&](bool writer_before_holder_reader) {
        topology t = slots_topology();
        t.add(activation("WW", input_info("S1"), activation_func::abs));
        t.add(activation("Z", input_info("S2"), activation_func::abs));
        auto prog = make_hand(eng, t, {"O", "WW", "Z"});
        prog->get_node("C").can_be_optimized(true);
        force_order(*prog, writer_before_holder_reader ? std::vector<std::string>{"S1", "S2", "C", "WW", "O", "Z"}
                                                         : std::vector<std::string>{"S1", "S2", "C", "O", "WW", "Z"});
        return prog;
    };
    {   // the reader Z of the sibling S2 is later than the writer: irrelevant, the siblings do not intersect. The reader O of the holder is earlier.
        auto prog = make(false);
        auto r = war_of(*prog, "WW", 0, {{"WW", 0}});
        EXPECT_VERDICT(r, war_verdict::Safe) << "static sibling slot pruned";
    }
    {   // the holder reader O is later than the writer: a victim
        auto prog = make(true);
        auto r = war_of(*prog, "WW", 0, {{"WW", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_EQ(r.res.witnesses.at(0).reader, &prog->get_node("O"));
        EXPECT_EQ(r.res.witnesses.at(0).via_node, &prog->get_node("C"));
    }
}

// W7: dynamic (RUNTIME offset) siblings are never pruned.
TEST(alias_validation, W7_runtime_offset_sibling_is_not_pruned) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}));
    t.add(input_layout("in2", layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}));
    t.add(activation("S1", input_info("in1"), activation_func::relu));
    t.add(activation("S2", input_info("in2"), activation_func::abs));
    t.add(concatenation("C", {input_info("S1"), input_info("S2")}, 1));
    t.add(activation("O", input_info("C"), activation_func::abs));
    t.add(activation("WW", input_info("S1"), activation_func::abs));
    t.add(activation("Z", input_info("S2"), activation_func::abs));
    auto prog = make_hand(eng, t, {"O", "WW", "Z"});
    auto& c = prog->get_node("C");
    ASSERT_TRUE(c.is_dynamic());
    c.can_be_optimized(true);
    c.set_runtime_skippable(true);
    force_order(*prog, {"S1", "S2", "C", "O", "WW", "Z"});
    // WW is a second user of S1, so the runtime concat would be cancelled by F9 and the real axis is not normalized: use the order as given
    std::unordered_map<const program_node*, uint32_t> pos;
    uint32_t k = 0;
    for (auto* n : prog->get_processing_order()) {
        if (!n->is_type<input_layout>()) {
            pos[n] = k++;
        }
    }
    const auto axis = exec_axis::for_test(exec_axis::status::ok, pos);
    auto r = war_of(*prog, "WW", 0, {{"WW", 0}}, &axis);
    EXPECT_VERDICT(r, war_verdict::Unsafe) << "the sibling reader Z is later and the byte ranges are only known at runtime";
    EXPECT_REASON(r, wr_late_reader);
    bool z = false;
    for (const auto& w : r.res.witnesses) {
        z = z || w.reader == &prog->get_node("Z");
    }
    EXPECT_TRUE(z) << why(r.res);
}

// W8: cascaded concat: overwriting an inner slot is visible in the outer holder.
TEST(alias_validation, W8_cascaded_concat_reaches_the_outer_holder) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", f32_l()));
    t.add(input_layout("in2", f32_l()));
    t.add(input_layout("in3", layout{ov::PartialShape{1, 8, 8, 8}, data_types::f32, format::bfyx}));
    t.add(activation("S1", input_info("in1"), activation_func::relu));
    t.add(activation("S2", input_info("in2"), activation_func::abs));
    t.add(concatenation("H1", {input_info("S1"), input_info("S2")}, 1, data_types::f32));
    t.add(activation("S3", input_info("in3"), activation_func::abs));
    t.add(concatenation("H2", {input_info("H1"), input_info("S3")}, 1, data_types::f32));
    t.add(activation("O", input_info("H2"), activation_func::abs));
    t.add(activation("WW", input_info("S1"), activation_func::abs));
    auto prog = make_hand(eng, t, {"O", "WW"});
    prog->get_node("H1").can_be_optimized(true);
    prog->get_node("H2").can_be_optimized(true);
    force_order(*prog, {"S1", "S2", "H1", "S3", "H2", "WW", "O"});
    auto r = war_of(*prog, "WW", 0, {{"WW", 0}});
    EXPECT_VERDICT(r, war_verdict::Unsafe);
    EXPECT_EQ(r.res.witnesses.at(0).reader, &prog->get_node("O"));
    EXPECT_EQ(r.res.witnesses.at(0).via_node, &prog->get_node("H2"));
    // writer after the reader of the outer holder: safe, the S3 sibling of H1 inside H2 is pruned
    force_order(*prog, {"S1", "S2", "H1", "S3", "H2", "O", "WW"});
    auto r2 = war_of(*prog, "WW", 0, {{"WW", 0}});
    EXPECT_VERDICT(r2, war_verdict::Safe);
}

// W9 (F11): the writer also reads the overwritten tensor (or a view of it) through another dependency.
TEST(alias_validation, W9_self_read_through_another_port_or_a_view) {
    auto& eng = get_test_engine();
    {   // eltwise(P, P): both dependencies are P
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(activation("P", input_info("in"), activation_func::relu));
        t.add(eltwise("W", {input_info("P"), input_info("P")}, eltwise_mode::sum));
        auto prog = make_hand(eng, t, {"W"});
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_self_read_other_port);
    }
    {   // eltwise(P, R) with R a cbo view of P
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(activation("P", input_info("in"), activation_func::relu));
        t.add(activation("R", input_info("P"), activation_func::abs));
        t.add(eltwise("W", {input_info("P"), input_info("R")}, eltwise_mode::sum));
        auto prog = make_hand(eng, t, {"W"});
        prog->get_node("R").can_be_optimized(true);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_self_read_other_port);
        // the same eltwise overwriting dependency 1 (the view): reads P through dependency 0
        auto r1 = war_of(*prog, "W", 1, {{"W", 1}});
        EXPECT_VERDICT(r1, war_verdict::Unsafe);
        EXPECT_REASON(r1, wr_self_read_other_port);
    }
}

// W10 (F6): external roots. Graph inputs, constants, and the like are never overwritten; the reason is not forgotten when the
// root is reached through an upstream in-place edge, and removing an upstream edge never turns Safe into Unsafe (W21).
TEST(alias_validation, W10_external_root_input_layout_and_data) {
    auto& eng = get_test_engine();
    {
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(activation("W", input_info("in"), activation_func::abs));
        auto prog = make_hand(eng, t, {"W"});
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_external_root);
        EXPECT_NO_REASON(r, wr_via_upstream_inplace);
    }
    {
        topology t;
        t.add(cldnn::data("d", get_test_engine().allocate_memory(f32_l())));
        t.add(activation("W", input_info("d"), activation_func::abs));
        auto prog = make_hand(eng, t, {"W"});
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_external_root);
    }
}

namespace {
topology ext_chain_topology() {
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("S1", input_info("in"), activation_func::abs));
    t.add(activation("S2", input_info("S1"), activation_func::abs));
    t.add(activation("S3", input_info("S2"), activation_func::abs));
    return t;
}
}  // namespace

TEST(alias_validation, W11_chain_from_external_root_marks_upstream_inplace_and_W21_is_monotonic) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, ext_chain_topology(), {"S3"});
    {
        auto r1 = war_of(*prog, "S1", 0, {{"S1", 0}, {"S2", 0}});
        EXPECT_VERDICT(r1, war_verdict::Unsafe);
        EXPECT_REASON(r1, wr_external_root);
        EXPECT_NO_REASON(r1, wr_via_upstream_inplace);
        auto r2 = war_of(*prog, "S2", 0, {{"S1", 0}, {"S2", 0}});
        EXPECT_VERDICT(r2, war_verdict::Unsafe);
        EXPECT_REASON(r2, wr_external_root);
        EXPECT_REASON(r2, wr_via_upstream_inplace);
        // without the upstream edge S2 writes into the buffer of S1, which is an ordinary allocation
        auto r3 = war_of(*prog, "S2", 0, {{"S2", 0}});
        EXPECT_VERDICT(r3, war_verdict::Safe);
    }
    // W21: for every subset of the upstream edges, a verdict of S3 can only get safer when edges are removed
    int verdict_of[4];
    for (int mask = 0; mask < 4; ++mask) {
        std::vector<std::pair<std::string, int>> hyp{{"S3", 0}};
        if (mask & 1) {
            hyp.push_back({"S1", 0});
        }
        if (mask & 2) {
            hyp.push_back({"S2", 0});
        }
        verdict_of[mask] = static_cast<int>(war_of(*prog, "S3", 0, hyp).res.verdict);
    }
    for (int sup = 0; sup < 4; ++sup) {
        for (int sub = 0; sub < 4; ++sub) {
            if ((sub & sup) == sub && verdict_of[sup] == static_cast<int>(war_verdict::Safe)) {
                EXPECT_EQ(verdict_of[sub], static_cast<int>(war_verdict::Safe)) << "sup=" << sup << " sub=" << sub;
            }
        }
    }
    EXPECT_EQ(verdict_of[0], static_cast<int>(war_verdict::Safe));
    EXPECT_EQ(verdict_of[3], static_cast<int>(war_verdict::Unsafe));
}

// W12 (R5a): a version of the overwritten tensor is a network output (or becomes one through mutable_data).
TEST(alias_validation, W12_visible_version) {
    auto& eng = get_test_engine();
    {
        auto prog = make_hand(eng, fan_topology(false), {"W", "P"});
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_visible_version);
    }
    {   // an output view of the overwritten tensor
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(activation("P", input_info("in"), activation_func::relu));
        t.add(activation("R", input_info("P"), activation_func::abs));
        t.add(activation("W", input_info("P"), activation_func::abs));
        auto prog = make_hand(eng, t, {"W", "R"});
        prog->get_node("R").can_be_optimized(true);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_visible_version);
    }
    {   // output through a mutable_data user
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(activation("P", input_info("in"), activation_func::relu));
        t.add(activation("W", input_info("P"), activation_func::abs));
        t.add(mutable_data("md", {input_info("P")}, eng.allocate_memory(f32_l())));
        auto prog = make_hand(eng, t, {"W", "md"});
        ASSERT_FALSE(prog->get_node("P").is_output());
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_visible_version);
    }
}

// W13: persistent writers: kv_cache, assign, stateless_kv
TEST(alias_validation, W13_kv_cache_and_assign_and_stateless_kv) {
    auto& eng = get_test_engine();
    {
        auto prog = make_hand(eng, kv_topology(false, "v0", "v0"), {"O"});
        auto aa = alias_analysis::build(*prog);
        const auto writers = enumerate_inplace_writers(aa);
        ASSERT_EQ(writers.size(), 1u);
        EXPECT_EQ(writers[0].writer, &prog->get_node("kv"));
        const auto ax = exec_axis::build(*prog);
        auto r = war_safe(aa, ax, writers[0]);
        EXPECT_EQ(r.verdict, war_verdict::Safe) << "registered persistent writer, no other reader of the variable alias\n" << why(r);
        // read_value only has CPU implementations: the member makes the verdict Unknown
        prog->get_node("rv").set_preferred_impl_type(impl_types::cpu);
        auto aa2 = alias_analysis::build(*prog);
        auto r2 = war_safe(aa2, ax, writers[0]);
        EXPECT_EQ(r2.verdict, war_verdict::Unknown) << why(r2);
        EXPECT_NE(r2.reasons & wr_cpu_member, 0u);
    }
    {   // an unregistered writer of a persistent root is a defect: a plain activation reading the read_value in place
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(read_value{"rv", {input_info("in")}, "v0", {f32_l()}});
        t.add(activation("W", input_info("rv"), activation_func::abs));
        auto prog = make_hand(eng, t, {"W"});
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_persistent_unregistered);
    }
    {   // assign: a copy into the variable, the version set starts at the aliases of the variable (N1), not at dependency 0
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(read_value{"rv", {input_info("in")}, "v0", {f32_l()}});
        t.add(activation("X", input_info("rv"), activation_func::abs));
        t.add(assign("as", {input_info("X")}, "v0", f32_l()));
        auto prog = make_hand(eng, t, {"as"});
        auto aa = alias_analysis::build(*prog);
        const auto ax = exec_axis::build(*prog);
        const auto writers = enumerate_inplace_writers(aa);
        ASSERT_EQ(writers.size(), 1u);
        auto r = war_safe(aa, ax, writers[0]);
        EXPECT_EQ(r.verdict, war_verdict::Safe) << why(r);
        prog->get_node("as").set_preferred_impl_type(impl_types::cpu);
        auto aa2 = alias_analysis::build(*prog);
        auto r2 = war_safe(aa2, ax, writers[0]);
        EXPECT_EQ(r2.verdict, war_verdict::Unknown) << why(r2);
        EXPECT_NE(r2.reasons & wr_cpu_writer, 0u);
        // a reader of the read_value that runs after the assign is a late reader of the variable alias
        topology t2 = [&] {
            topology u;
            u.add(input_layout("in", f32_l()));
            u.add(read_value{"rv", {input_info("in")}, "v0", {f32_l()}});
            u.add(activation("X", input_info("rv"), activation_func::abs));
            u.add(assign("as", {input_info("X")}, "v0", f32_l()));
            u.add(activation("late", input_info("rv"), activation_func::abs));
            return u;
        }();
        auto prog2 = make_hand(eng, t2, {"as", "late"});
        force_order(*prog2, {"rv", "X", "as", "late"});
        auto aa3 = alias_analysis::build(*prog2);
        const auto ax3 = exec_axis::build(*prog2);
        auto r3 = war_safe(aa3, ax3, enumerate_inplace_writers(aa3).at(0));
        EXPECT_EQ(r3.verdict, war_verdict::Unsafe) << why(r3);
        EXPECT_NE(r3.reasons & wr_late_reader, 0u);
    }
    {   // stateless_kv: runtime guarded, registered on `past`
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
        const auto ax = exec_axis::build(*prog);
        const auto writers = enumerate_inplace_writers(aa);
        ASSERT_EQ(writers.size(), 1u);
        auto r = war_safe(aa, ax, writers[0]);
        EXPECT_EQ(r.verdict, war_verdict::Unknown) << why(r);
        EXPECT_NE(r.reasons & wr_runtime_guarded, 0u);
        EXPECT_EQ(r.reasons & wr_external_root, 0u) << "past is the registered target of this writer";
    }
}

// W14: LoRA is opaque.
TEST(alias_validation, W14_lora_is_unknown_opaque) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, lora_topology(), {"O"});
    auto aa = alias_analysis::build(*prog);
    const auto ax = exec_axis::build(*prog);
    const auto writers = enumerate_inplace_writers(aa);
    ASSERT_EQ(writers.size(), 1u);
    auto r = war_safe(aa, ax, writers[0]);
    EXPECT_EQ(r.verdict, war_verdict::Unknown) << why(r);
    EXPECT_NE(r.reasons & wr_opaque, 0u);
}

// W15: OOOQ: only the order independent evidence survives.
TEST(alias_validation, W15_out_of_order_queue) {
    auto& eng = get_test_engine();
    {
        auto prog = make_hand(eng, fan_topology(), fan_outs(), QueueTypes::out_of_order);
        force_order(*prog, {"P", "W", "Y"});
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unknown);
        EXPECT_REASON(r, wr_oooq);
        EXPECT_NO_REASON(r, wr_late_reader);
    }
    {
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(activation("W", input_info("in"), activation_func::abs));
        auto prog = make_hand(eng, t, {"W"}, QueueTypes::out_of_order);
        ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::out_of_order);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe) << "the external root is order independent evidence";
        EXPECT_REASON(r, wr_oooq);
        EXPECT_REASON(r, wr_external_root);
    }
}

// W16: CPU and shape_of
TEST(alias_validation, W16_cpu_shape_of_and_body) {
    auto& eng = get_test_engine();
    {   // CPU reader: Unknown, a GPU late reader on top makes it Unsafe
        auto prog = make_hand(eng, fan_topology(true, true), fan_outs(true, true));
        prog->get_node("Y").set_preferred_impl_type(impl_types::cpu);
        force_order(*prog, {"P", "Y", "W", "Z"});
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        EXPECT_REASON(r, wr_cpu_reader);
        EXPECT_REASON(r, wr_late_reader);
        force_order(*prog, {"P", "Z", "Y", "W"});
        auto r2 = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r2, war_verdict::Unknown);
        EXPECT_REASON(r2, wr_cpu_reader);
    }
    {   // CPU writer
        auto prog = make_hand(eng, fan_topology(false), fan_outs(false));
        prog->get_node("W").set_preferred_impl_type(impl_types::cpu);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unknown);
        EXPECT_REASON(r, wr_cpu_writer);
    }
    {   // shape_of reader is skipped
        topology t = fan_topology(false);
        t.add(shape_of("S", input_info("P"), data_types::i32));
        auto prog = make_hand(eng, t, {"W", "S"});
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Safe);
        EXPECT_REASON(r, wr_no_other_reader);
    }
    {   // member / reader in a shape_of subgraph
        auto prog = make_hand(eng, fan_topology(), fan_outs());
        prog->get_node("P").set_in_shape_of_subgraph(true);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unknown);
        EXPECT_REASON(r, wr_shape_of_member);
        auto prog2 = make_hand(eng, fan_topology(), fan_outs());
        prog2->get_node("Y").set_in_shape_of_subgraph(true);
        auto r2 = war_of(*prog2, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r2, war_verdict::Unknown);
        EXPECT_REASON(r2, wr_shape_of_subgraph_reader);
    }
    {   // body / internal programs
        auto body = make_hand(eng, fan_topology(false), fan_outs(false), QueueTypes::in_order, false, true);
        auto r = war_of(*body, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unknown);
        EXPECT_REASON(r, wr_body_or_internal);
        auto internal = make_hand(eng, fan_topology(false), fan_outs(false), QueueTypes::in_order, true, false);
        auto r2 = war_of(*internal, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r2, war_verdict::Unknown);
        EXPECT_REASON(r2, wr_body_or_internal);
    }
}

// W17: dynamic nodes never get a MUST in-place edge; a hypothetical writer on a dynamic node is MAYBE / MAY_INPLACE.
TEST(alias_validation, W17_dynamic_node_never_has_must_inplace) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(activation("W", input_info("P"), activation_func::abs));
    auto prog = make_hand(eng, t, {"W"});
    ASSERT_TRUE(prog->get_node("W").is_dynamic());
    EXPECT_TRUE(alias_analysis::build(*prog).edges().empty()) << "no sum can be formed on dynamic nodes";
    std::vector<hypothetical_inplace> extra{{&prog->get_node("W"), 0}};
    EXPECT_TRUE(alias_analysis::build(*prog).rejected_hypotheticals().empty());
    auto aa = alias_analysis::build(*prog, compile_time_envelope{}, &extra);
    ASSERT_EQ(aa.edges().size(), 1u);
    EXPECT_EQ(aa.edges()[0].kind, alias_kind::MAY_INPLACE);
    EXPECT_EQ(aa.edges()[0].act, edge_activity::MAYBE);
}

// W18: positions come from the axis, nothing else (W19): the same graph with an axis that swaps two nodes flips the verdict.
TEST(alias_validation, W18_positions_are_read_from_the_axis_only) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, fan_topology(), fan_outs());
    force_order(*prog, {"P", "W", "Y"});
    const auto real = exec_axis::build(*prog);
    ASSERT_TRUE(real.valid());
    std::unordered_map<const program_node*, uint32_t> pos;
    for (auto* n : prog->get_processing_order()) {
        if (real.position(*n)) {
            pos[n] = *real.position(*n);
        }
    }
    std::swap(pos[&prog->get_node("W")], pos[&prog->get_node("Y")]);
    const auto swapped = exec_axis::for_test(exec_axis::status::ok, pos);
    auto r_real = war_of(*prog, "W", 0, {{"W", 0}}, &real);
    auto r_swap = war_of(*prog, "W", 0, {{"W", 0}}, &swapped);
    EXPECT_VERDICT(r_real, war_verdict::Unsafe);
    EXPECT_VERDICT(r_swap, war_verdict::Safe);
    // a reader without a position (not executable / unknown to the axis) is reported as such
    std::unordered_map<const program_node*, uint32_t> only_w{{&prog->get_node("W"), 3}};
    const auto partial = exec_axis::for_test(exec_axis::status::ok, only_w);
    auto r_partial = war_of(*prog, "W", 0, {{"W", 0}}, &partial);
    EXPECT_VERDICT(r_partial, war_verdict::Unknown);
    EXPECT_REASON(r_partial, wr_reader_off_axis);
}

// W20: converging readers A -> X -> B, A -> u, B -> u: readers are enumerated per dependency index through the shared helper.
TEST(alias_validation, W20_converging_readers) {
    auto& eng = get_test_engine();
    auto make = [&](bool writer_last) {
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(activation("A", input_info("in"), activation_func::relu));
        t.add(activation("X", input_info("A"), activation_func::abs));
        t.add(activation("B", input_info("X"), activation_func::abs));
        t.add(eltwise("u", {input_info("A"), input_info("B")}, eltwise_mode::sum));
        t.add(activation("W", input_info("A"), activation_func::abs));
        auto prog = make_hand(eng, t, {"u", "W"});
        force_order(*prog, writer_last ? std::vector<std::string>{"A", "X", "B", "u", "W"} : std::vector<std::string>{"A", "W", "X", "B", "u"});
        return prog;
    };
    {
        auto prog = make(true);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Safe);
        EXPECT_NO_REASON(r, wr_no_other_reader);
    }
    {
        auto prog = make(false);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe);
        std::set<std::string> readers;
        for (const auto& w : r.res.witnesses) {
            readers.insert(w.reader->id());
        }
        EXPECT_TRUE(readers.count("X") && readers.count("u")) << why(r.res);
    }
}

// W22 (B2): a tensor that is a STATIC_BYTES view of P and at the same time a slot of a static holder H. The byte relation of the
// holder's other slots to P is unknown, the static sibling pruning must not apply.
TEST(alias_validation, W22_view_that_is_also_a_slot_does_not_allow_sibling_pruning) {
    auto& eng = get_test_engine();
    auto make = [&](bool as_view) {
        topology t;
        t.add(input_layout("in", f32_l(ov::PartialShape{1, 16, 8, 8})));
        t.add(input_layout("in2", f32_l(ov::PartialShape{1, 8, 8, 8})));
        t.add(activation("P", input_info("in"), activation_func::relu));
        t.add(crop("K", input_info("P"), tensor(1, 8, 8, 8), tensor(0, 0, 0, 0)));
        t.add(activation("S2", input_info("in2"), activation_func::abs));
        t.add(concatenation("H", {input_info("K"), input_info("S2")}, 1, data_types::f32));
        t.add(activation("OH", input_info("H"), activation_func::abs));
        t.add(activation("Z", input_info("S2"), activation_func::abs));
        t.add(activation("W", input_info("P"), activation_func::abs));
        auto prog = make_hand(eng, t, {"OH", "Z", "W"});
        prog->get_node("H").can_be_optimized(true);
        prog->get_node("K").can_be_optimized(as_view);   // as a pure slot (not a view) the pruning is legal
        force_order(*prog, {"P", "K", "S2", "H", "OH", "W", "Z"});
        return prog;
    };
    {
        auto prog = make(true);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_NE(r.res.verdict, war_verdict::Safe) << why(r.res);
        EXPECT_REASON(r, wr_late_reader);
        EXPECT_EQ(r.res.witnesses.at(0).reader, &prog->get_node("Z"));
    }
    {   // control: K is not a view, P is not related to the holder at all
        auto prog = make(false);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_NO_REASON(r, wr_late_reader);
    }
}

// Pattern C (documented, no oneDNN needed): a residual chain of 25 blocks, every sum writing in place into the previous residual stream.
TEST(alias_validation, WC_residual_chain_of_25_hops_is_safe_for_every_writer) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in", f32_l()));
    t.add(activation("S0", input_info("in"), activation_func::relu));
    std::vector<std::pair<std::string, int>> hyp;
    for (int i = 1; i <= 25; ++i) {
        const std::string prev = "S" + std::to_string(i - 1);
        t.add(activation("f" + std::to_string(i), input_info(prev), activation_func::abs));
        t.add(eltwise("S" + std::to_string(i), {input_info("f" + std::to_string(i)), input_info(prev)}, eltwise_mode::sum));
        hyp.push_back({"S" + std::to_string(i), 1});
    }
    auto prog = make_hand(eng, t, {"S25"});
    std::vector<hypothetical_inplace> extra;
    for (const auto& h : hyp) {
        extra.push_back({&prog->get_node(h.first), h.second});
    }
    auto aa = alias_analysis::build(*prog, compile_time_envelope{}, &extra);
    ASSERT_TRUE(aa.rejected_hypotheticals().empty());
    const auto ax = exec_axis::build(*prog);
    size_t safe = 0;
    for (const auto& h : hyp) {
        auto r = war_safe(aa, ax, war_query{&prog->get_node(h.first), h.second, 0});
        safe += r.verdict == war_verdict::Safe ? 1 : 0;
        EXPECT_EQ(r.verdict, war_verdict::Safe) << h.first << " " << why(r);
    }
    EXPECT_EQ(safe, 25u);
    // a late reader of one residual stream makes exactly its writer unsafe
    topology t2 = t;
    t2.add(activation("late", input_info("S10"), activation_func::abs));
    t2.add(activation("fin", input_info("S25"), activation_func::abs));
    auto prog2 = make_hand(eng, t2, {"fin", "late"});
    force_order(*prog2, {"late", "fin"});   // fin depends on S25: late runs after S25
    std::vector<hypothetical_inplace> extra2;
    for (const auto& h : hyp) {
        extra2.push_back({&prog2->get_node(h.first), h.second});
    }
    auto aa2 = alias_analysis::build(*prog2, compile_time_envelope{}, &extra2);
    const auto ax2 = exec_axis::build(*prog2);
    // S11 overwrites S10, and "late" reads S10 after S25: unsafe
    auto r11 = war_safe(aa2, ax2, war_query{&prog2->get_node("S11"), 1, 0});
    EXPECT_EQ(r11.verdict, war_verdict::Unsafe) << why(r11);
    auto r5 = war_safe(aa2, ax2, war_query{&prog2->get_node("S5"), 1, 0});
    EXPECT_EQ(r5.verdict, war_verdict::Safe) << why(r5);
}

// ==============================================================================================================================
// R*. compile time rules
// ==============================================================================================================================
namespace {
alias_validation_report validate(program& prog, const std::vector<hypothetical_inplace>* extra = nullptr) {
    auto aa = alias_analysis::build(prog, compile_time_envelope{}, extra);
    return validate_alias_rules(prog, aa);
}
bool has_hit(const alias_validation_report& r, rule_id id, severity sev, const std::string& cell_part, const std::string& node = "") {
    for (const auto& h : r.hits) {
        if (h.id == id && h.sev == sev && h.cell.find(cell_part) != std::string::npos && (node.empty() || (h.a && h.a->id() == node))) {
            return true;
        }
    }
    return false;
}
std::string hits_str(const alias_validation_report& r) {
    std::ostringstream os;
    for (const auto& h : r.hits) {
        os << to_string(h.id) << (h.sev == severity::defect ? "/D " : h.sev == severity::war ? "/W " : "/I ") << (h.a ? h.a->id() : "?") << " " << h.cell << "\n";
    }
    return os.str();
}
}  // namespace

TEST(alias_validation, R1_legal_graphs_have_no_defects) {
    auto& eng = get_test_engine();
    for (const bool cbo : {true, false}) {
        auto p = make_hand(eng, chain_topology(), {"O"});
        p->get_node("R").can_be_optimized(cbo);
        p->get_node("Q").can_be_optimized(cbo);
        auto rep = validate(*p);
        EXPECT_TRUE(rep.ran);
        EXPECT_EQ(rep.defect_count(), 0u) << hits_str(rep);
    }
    {
        auto p = make_hand(eng, concat_topology(), {"O"});
        p->get_node("C").can_be_optimized(true);
        EXPECT_EQ(validate(*p).defect_count(), 0u);
    }
    {
        auto p = make_hand(eng, crop_topology(), {"O"});
        p->get_node("K").can_be_optimized(true);
        EXPECT_EQ(validate(*p).defect_count(), 0u);
    }
    {   // static cascade on the same axis is legal
        topology t;
        t.add(input_layout("in1", f32_l()));
        t.add(input_layout("in2", f32_l()));
        t.add(input_layout("in3", layout{ov::PartialShape{1, 8, 8, 8}, data_types::f32, format::bfyx}));
        t.add(activation("S1", input_info("in1"), activation_func::relu));
        t.add(activation("S2", input_info("in2"), activation_func::abs));
        t.add(concatenation("H1", {input_info("S1"), input_info("S2")}, 1, data_types::f32));
        t.add(activation("S3", input_info("in3"), activation_func::abs));
        t.add(concatenation("H2", {input_info("H1"), input_info("S3")}, 1, data_types::f32));
        t.add(activation("O", input_info("H2"), activation_func::abs));
        auto p = make_hand(eng, t, {"O"});
        p->get_node("H1").can_be_optimized(true);
        p->get_node("H2").can_be_optimized(true);
        auto rep = validate(*p);
        EXPECT_EQ(rep.defect_count(), 0u) << hits_str(rep);
    }
    {   // dynamic runtime in-place concat: MAY edges only: info, never a defect
        auto p = make_hand(eng, dyn_concat_topology(), {"O"});
        p->get_node("C").can_be_optimized(true);
        p->get_node("C").set_runtime_skippable(true);
        p->get_node("R").set_runtime_skippable(true);
        EXPECT_EQ(validate(*p).defect_count(), 0u);
    }
    for (const bool cbo : {true, false}) {   // kv_cache: the appended output is larger than the variable, which is legal state aliasing (V1 info)
        auto p = make_hand(eng, kv_topology(false, "v0", "v0"), {"O"});
        p->get_node("kv").can_be_optimized(cbo);
        auto rep = validate(*p);
        EXPECT_EQ(rep.defect_count(), 0u) << hits_str(rep);
    }
    {
        auto rep = validate(*make_hand(eng, lora_topology(), {"O"}));
        EXPECT_EQ(rep.defect_count(), 0u) << hits_str(rep);
        EXPECT_TRUE(has_hit(rep, rule_id::R2, severity::info, "opaque"));
    }
}

TEST(alias_validation, R1_defect_cells_are_detected) {
    auto& eng = get_test_engine();
    {   // VIEW x SLOT (static): a concat predecessor that is itself a view
        auto p = make_hand(eng, concat_topology(), {"O"});
        p->get_node("C").can_be_optimized(true);
        p->get_node("P1").can_be_optimized(true);
        auto rep = validate(*p);
        EXPECT_TRUE(has_hit(rep, rule_id::R1, severity::defect, "VIEW x SLOT", "P1")) << hits_str(rep);
        EXPECT_NE(rep.count(rule_id::R2, severity::info), 0u) << "also an R2 role conflict statistic";
    }
    {   // VIEW x SLOT with a dynamic holder is a MAY relation (F24): info
        auto p = make_hand(eng, dyn_concat_topology(), {"O"});
        p->get_node("C").can_be_optimized(true);
        p->get_node("C").set_runtime_skippable(true);
        p->get_node("P1").set_runtime_skippable(true);
        auto rep = validate(*p);
        EXPECT_TRUE(has_hit(rep, rule_id::R1, severity::info, "VIEW x SLOT", "P1")) << hits_str(rep);
        EXPECT_EQ(rep.defect_count(), 0u) << hits_str(rep);
    }
    {   // SLOT x OUT
        auto p = make_hand(eng, concat_topology(), {"O", "P1"});
        p->get_node("C").can_be_optimized(true);
        auto rep = validate(*p);
        EXPECT_TRUE(has_hit(rep, rule_id::R1, severity::defect, "SLOT x OUT", "P1")) << hits_str(rep);
    }
    {   // HOLDER x OUT
        auto p = make_hand(eng, concat_topology(), {"O", "C"});
        p->get_node("C").can_be_optimized(true);
        auto rep = validate(*p);
        EXPECT_TRUE(has_hit(rep, rule_id::R1, severity::defect, "HOLDER x OUT", "C")) << hits_str(rep);
    }
    {   // SLOT x HOLDER on different axes
        topology t;
        t.add(input_layout("in1", f32_l()));
        t.add(input_layout("in2", f32_l()));
        t.add(input_layout("in3", layout{ov::PartialShape{1, 8, 8, 8}, data_types::f32, format::bfyx}));
        t.add(activation("S1", input_info("in1"), activation_func::relu));
        t.add(activation("S2", input_info("in2"), activation_func::abs));
        t.add(concatenation("H1", {input_info("S1"), input_info("S2")}, 1, data_types::f32));
        t.add(activation("S3", input_info("in3"), activation_func::abs));
        t.add(concatenation("H2", {input_info("H1"), input_info("S3")}, 2, data_types::f32));
        t.add(activation("O", input_info("H2"), activation_func::abs));
        auto p = make_hand(eng, t, {"O"});
        p->get_node("H1").can_be_optimized(true);
        p->get_node("H2").can_be_optimized(true);
        auto rep = validate(*p);
        EXPECT_TRUE(has_hit(rep, rule_id::R1, severity::defect, "SLOT x HOLDER", "H1")) << hits_str(rep);
    }
    {   // SLOT x PERSIST (A18g, constructed by hand)
        topology t = kv_topology(false, "v0", "v0");
        t.add(input_layout("other", layout{ov::PartialShape{1, 2, 5, 8}, data_types::f16, format::bfyx}));
        t.add(concatenation("C", {input_info("kv"), input_info("other")}, 2));
        t.add(activation("O2", input_info("C"), activation_func::abs));
        auto p = make_hand(eng, t, {"O", "O2"});
        p->get_node("C").can_be_optimized(true);
        auto rep = validate(*p);
        EXPECT_TRUE(has_hit(rep, rule_id::R1, severity::defect, "SLOT x PERSIST", "kv")) << hits_str(rep);
    }
    {   // SLOT x SOC: shape_of subgraph member as concat predecessor is a defect, a CPU one is info
        auto p = make_hand(eng, concat_topology(), {"O"});
        p->get_node("C").can_be_optimized(true);
        p->get_node("P1").set_in_shape_of_subgraph(true);
        p->get_node("P2").set_preferred_impl_type(impl_types::cpu);
        auto rep = validate(*p);
        EXPECT_TRUE(has_hit(rep, rule_id::R1, severity::defect, "SLOT x SOC (shape_of)", "P1")) << hits_str(rep);
        EXPECT_TRUE(has_hit(rep, rule_id::R1, severity::info, "SLOT x SOC (cpu)", "P2")) << hits_str(rep);
    }
}

TEST(alias_validation, R5b_in_place_writer_that_is_or_leads_to_an_output) {
    auto& eng = get_test_engine();
    auto build = [&](const std::vector<std::string>& outs, bool view_of_w_cbo) {
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(activation("P", input_info("in"), activation_func::relu));
        t.add(activation("W", input_info("P"), activation_func::abs));
        t.add(activation("R", input_info("W"), activation_func::abs));
        t.add(activation("OR", input_info("R"), activation_func::abs));
        auto prog = make_hand(eng, t, outs);
        prog->get_node("R").can_be_optimized(view_of_w_cbo);
        return prog;
    };
    auto run = [&](program& prog) {
        std::vector<hypothetical_inplace> extra{{&prog.get_node("W"), 0}};
        return validate(prog, &extra);
    };
    {   // control: neither W nor its forward view chain is visible
        auto prog = build({"OR"}, true);
        auto rep = run(*prog);
        EXPECT_EQ(rep.count(rule_id::R5b, severity::defect), 0u) << hits_str(rep);
    }
    {   // W itself is an output
        auto prog = build({"W", "OR"}, true);
        auto rep = run(*prog);
        EXPECT_TRUE(has_hit(rep, rule_id::R5b, severity::defect, "INPLACE_W x OUT", "W")) << hits_str(rep);
    }
    {   // a view of W is an output
        auto prog = build({"R", "OR"}, true);
        auto rep = run(*prog);
        EXPECT_TRUE(has_hit(rep, rule_id::R5b, severity::defect, "forward view chain", "W")) << hits_str(rep);
    }
    {   // R is an output but not a view: not visible
        auto prog = build({"R", "OR"}, false);
        auto rep = run(*prog);
        EXPECT_EQ(rep.count(rule_id::R5b, severity::defect), 0u) << hits_str(rep);
    }
}

TEST(alias_validation, R5c_and_V1_and_R2_counts) {
    auto& eng = get_test_engine();
    {   // mutable_data propagation
        topology t;
        t.add(input_layout("in", f32_l()));
        t.add(activation("D", input_info("in"), activation_func::relu));
        t.add(activation("X", input_info("D"), activation_func::abs));
        t.add(mutable_data("md", {input_info("D")}, eng.allocate_memory(f32_l())));
        auto prog = make_hand(eng, t, {"X", "md"});
        auto rep = validate(*prog);
        EXPECT_TRUE(has_hit(rep, rule_id::R5c, severity::info, "mutable_data", "D")) << hits_str(rep);
        EXPECT_EQ(rep.count(rule_id::R5c, severity::info), 1u);
    }
    {   // V1: member bytes above root bytes with a MUST edge: defect (A15)
        topology t;
        t.add(input_layout("in", layout{ov::PartialShape{1, 4, 8, 8}, data_types::f16, format::bfyx}));
        t.add(activation("P", input_info("in"), activation_func::relu));
        t.add(reorder("R", input_info("P"), format::bfyx, data_types::f32));
        t.add(activation("O", input_info("R"), activation_func::abs));
        auto prog = make_hand(eng, t, {"O"});
        prog->get_node("R").can_be_optimized(true);
        auto rep = validate(*prog);
        EXPECT_EQ(rep.count(rule_id::V1, severity::defect), 1u) << hits_str(rep);
        EXPECT_TRUE(has_hit(rep, rule_id::R2, severity::info, "member_exceeds_root"));
        // the same with a MAY edge: information only
        topology t2 = t;
        auto prog2 = make_hand(eng, t2, {"O"});
        prog2->get_node("R").set_runtime_skippable(true);
        auto rep2 = validate(*prog2);
        EXPECT_EQ(rep2.count(rule_id::V1, severity::defect), 0u) << hits_str(rep2);
    }
    {   // multi edge tensors are listed (A18g: concat slot plus kv alias)
        topology t = kv_topology(false, "v0", "v0");
        t.add(input_layout("other", layout{ov::PartialShape{1, 2, 5, 8}, data_types::f16, format::bfyx}));
        t.add(concatenation("C", {input_info("kv"), input_info("other")}, 2));
        t.add(activation("O2", input_info("C"), activation_func::abs));
        auto prog = make_hand(eng, t, {"O", "O2"});
        prog->get_node("C").can_be_optimized(true);
        auto rep = validate(*prog);
        EXPECT_TRUE(has_hit(rep, rule_id::R2, severity::info, "multi_edge", "kv")) << hits_str(rep);
        EXPECT_TRUE(has_hit(rep, rule_id::R2, severity::info, "allocators_not_one")) << hits_str(rep);
    }
}

// war report: every in-place writer is listed, including the origin and the verdict
TEST(alias_validation, war_report_lists_every_in_place_writer) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, ext_chain_topology(), {"S3"});
    std::vector<hypothetical_inplace> extra{{&prog->get_node("S2"), 0}, {&prog->get_node("S3"), 0}};
    auto rep = validate(*prog, &extra);
    ASSERT_EQ(rep.war.size(), 2u);
    for (const auto& w : rep.war) {
        EXPECT_EQ(w.origin, edge_origin::hypothetical);
        EXPECT_EQ(w.kind, alias_kind::MUST_INPLACE);
        EXPECT_EQ(w.r.verdict, war_verdict::Safe);
    }
    EXPECT_EQ(rep.war_count(war_verdict::Safe), 2u);
    EXPECT_FALSE(rep.summary().empty());
}

// wiring in prepare_memory_dependencies, and the switch
TEST(alias_validation, program_wiring_and_switch) {
    auto& eng = get_test_engine();
    auto topo = [] {
        topology t;
        t.add(input_layout("in1", layout{ov::PartialShape{1, 16, 8, 8}, data_types::f32, format::bfyx}));
        t.add(input_layout("in2", layout{ov::PartialShape{1, 16, 8, 8}, data_types::f32, format::bfyx}));
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(reorder("out", input_info("C"), format::bfyx, data_types::f32));
        return t;
    };
    for (const bool on : {true, false}) {
        ExecutionConfig cfg = get_test_default_config(eng);
        cfg.set_property(ov::intel_gpu::optimize_data(true));
        cfg.set_property(ov::intel_gpu::enable_alias_shadow_validation(on));
        auto prog = program::build_program(eng, topo(), cfg, false, false);
        if (!on) {
            EXPECT_EQ(prog->get_alias_validation_report(), nullptr);
            continue;
        }
        const auto* rep = prog->get_alias_validation_report();
        ASSERT_NE(rep, nullptr);
        EXPECT_TRUE(rep->ran) << rep->skipped_reason;
        EXPECT_EQ(rep->defect_count(), 0u) << hits_str(*rep);
        EXPECT_EQ(rep->axis.get_status(), prog->get_config().get_queue_type() == QueueTypes::in_order ? exec_axis::status::ok : exec_axis::status::not_in_order);
    }
    {   // internal and body programs are skipped with a reason (report exists because the switch is on)
        ExecutionConfig cfg = get_test_default_config(eng);
        cfg.set_property(ov::intel_gpu::enable_alias_shadow_validation(true));
        auto body = program::build_program(eng, topo(), cfg, false, false, /*is_body_program=*/true);
        const auto* body_rep = body->get_alias_validation_report();
        ASSERT_NE(body_rep, nullptr) << "a body program must carry a (skipped) report when the switch is on";
        EXPECT_FALSE(body_rep->ran);
        EXPECT_EQ(body_rep->skipped_reason, "body");
        auto internal = program::build_program(eng, topo(), cfg, /*is_internal=*/true, false);
        const auto* internal_rep = internal->get_alias_validation_report();
        ASSERT_NE(internal_rep, nullptr) << "an internal program must carry a (skipped) report when the switch is on";
        EXPECT_FALSE(internal_rep->ran);
        EXPECT_EQ(internal_rep->skipped_reason, "internal");
    }
}


// ==============================================================================================================================
// P0.5b-4: X1 (COVERING_VIA_SLOT) and hardening
// ==============================================================================================================================
namespace {
// S1, S2 -> C (static in-place holder). K = crop(C) covering the bytes of S1 (or crop(R) with R a ZERO view of C).
// W writes K in place, Z is a second reader of S1 that runs after W, O (reader of C) runs before W.
program::ptr x1_graph(engine& eng, bool via_reshape, bool z_late) {
    topology t;
    t.add(input_layout("in1", f32_l()));
    t.add(input_layout("in2", f32_l()));
    t.add(activation("S1", input_info("in1"), activation_func::relu));
    t.add(activation("S2", input_info("in2"), activation_func::abs));
    t.add(concatenation("C", {input_info("S1"), input_info("S2")}, 1, data_types::f32));
    t.add(activation("O", input_info("C"), activation_func::abs));
    if (via_reshape) {
        t.add(activation("R", input_info("C"), activation_func::abs));
        t.add(crop("K", input_info("R"), tensor(1, 4, 8, 8), tensor(0, 0, 0, 0)));
    } else {
        t.add(crop("K", input_info("C"), tensor(1, 4, 8, 8), tensor(0, 0, 0, 0)));
    }
    t.add(activation("W", input_info("K"), activation_func::abs));
    t.add(activation("Z", input_info("S1"), activation_func::abs));
    auto prog = make_hand(eng, t, {"O", "W", "Z"});
    prog->get_node("C").can_be_optimized(true);
    prog->get_node("K").can_be_optimized(true);
    if (via_reshape) {
        prog->get_node("R").can_be_optimized(true);
        force_order(*prog, z_late ? std::vector<std::string>{"S1", "S2", "C", "O", "R", "K", "W", "Z"} : std::vector<std::string>{"S1", "S2", "C", "O", "R", "K", "Z", "W"});
    } else {
        force_order(*prog, z_late ? std::vector<std::string>{"S1", "S2", "C", "O", "K", "W", "Z"} : std::vector<std::string>{"S1", "S2", "C", "O", "K", "Z", "W"});
    }
    return prog;
}
}  // namespace

TEST(alias_validation, W23_crop_of_a_static_holder_written_in_place_sees_late_slot_readers) {
    auto& eng = get_test_engine();
    for (const bool via_reshape : {false, true}) {   // W23 and W23b (crop o view)
        auto prog = x1_graph(eng, via_reshape, true);
        auto r = war_of(*prog, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r, war_verdict::Unsafe) << "via_reshape=" << via_reshape;
        EXPECT_REASON(r, wr_late_reader);
        bool z = false;
        for (const auto& w : r.res.witnesses) {
            z = z || w.reader == &prog->get_node("Z");
        }
        EXPECT_TRUE(z) << "via_reshape=" << via_reshape << " " << why(r.res);
        // control: Z before the writer: nothing late
        auto ok = x1_graph(eng, via_reshape, false);
        auto r2 = war_of(*ok, "W", 0, {{"W", 0}});
        EXPECT_VERDICT(r2, war_verdict::Safe) << "via_reshape=" << via_reshape;
    }
}

TEST(alias_validation, W24_writer_without_a_position_is_never_safe) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, fan_topology(false), fan_outs(false));
    std::unordered_map<const program_node*, uint32_t> pos{{&prog->get_node("P"), 0}};   // W is missing from the axis
    const auto axis = exec_axis::for_test(exec_axis::status::ok, pos);
    auto r = war_of(*prog, "W", 0, {{"W", 0}}, &axis);
    EXPECT_VERDICT(r, war_verdict::Unknown);
    EXPECT_REASON(r, wr_writer_off_axis);
    // with the writer on the axis the same graph is Safe
    auto ok = war_of(*prog, "W", 0, {{"W", 0}});
    EXPECT_VERDICT(ok, war_verdict::Safe);
}

TEST(alias_validation, W25_cbo_kv_cache_is_enumerated_as_an_in_place_writer) {
    auto& eng = get_test_engine();
    auto prog = make_hand(eng, kv_topology(false, "v0", "v0"), {"O"});
    prog->get_node("kv").can_be_optimized(true);   // only a MUST_VIEW edge, no in-place edge
    prog->get_node("rv").set_preferred_impl_type(impl_types::cpu);
    auto aa = alias_analysis::build(*prog);
    for (const auto& e : aa.edges()) {
        EXPECT_NE(e.kind, alias_kind::MAY_INPLACE) << "precondition: no in-place edge for the cbo kv_cache";
    }
    const auto writers = enumerate_inplace_writers(aa);
    ASSERT_EQ(writers.size(), 1u);
    EXPECT_EQ(writers[0].writer, &prog->get_node("kv"));
    auto rep = validate_alias_rules(*prog, aa);
    ASSERT_EQ(rep.war.size(), 1u);
    EXPECT_EQ(rep.war[0].r.verdict, war_verdict::Unknown) << why(rep.war[0].r);
    EXPECT_NE(rep.war[0].r.reasons & wr_cpu_member, 0u);
}

TEST(alias_validation, V1_persistent_class_downgrade_only_when_the_biggest_member_is_persistent) {
    auto& eng = get_test_engine();
    auto build = [&](bool with_big_view) {
        topology t = kv_topology(false, "v0", "v0");
        if (with_big_view) {
            t.add(reorder("RR", input_info("kv"), format::bfyx, data_types::f32));   // twice the bytes of the f16 kv output: a non persistent member
        }
        auto prog = make_hand(eng, t, with_big_view ? std::vector<std::string>{"O", "RR"} : std::vector<std::string>{"O"});
        prog->get_node("kv").can_be_optimized(true);
        if (with_big_view) {
            prog->get_node("RR").can_be_optimized(true);
        }
        return prog;
    };
    {   // biggest member is the kv_cache itself (legal state aliasing): info
        auto prog = build(false);
        auto rep = validate(*prog);
        EXPECT_EQ(rep.count(rule_id::V1, severity::defect), 0u) << hits_str(rep);
        EXPECT_GE(rep.count(rule_id::V1, severity::info), 1u) << hits_str(rep);
    }
    {   // a non persistent MUST member beyond the root: defect
        auto prog = build(true);
        auto rep = validate(*prog);
        EXPECT_EQ(rep.count(rule_id::V1, severity::defect), 1u) << hits_str(rep);
    }
}
