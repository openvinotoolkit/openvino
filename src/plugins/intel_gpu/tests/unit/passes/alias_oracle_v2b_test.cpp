// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// P2-a S3: V2b, the runtime lifetime check on the memory actually bound (design 19, C6). Report only.
// The V2b verdict must not depend on alias_analysis, node flags or class_lifetimes: the independence test below builds a clobber that
// the lifetime engine cannot know about and checks that V2b still reports it.

#include "test_utils.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "alias_analysis.h"
#include "alias_oracle.h"
#include "class_lifetime.h"
#include "exec_axis.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/selective_ssm.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "primitive_inst.h"
#include "program_node.h"

using namespace cldnn;
using namespace ::tests;

namespace {

ExecutionConfig make_cfg(engine& eng, bool oracle = true, bool optimize = true) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(optimize));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    config.set_property(ov::intel_gpu::enable_alias_shadow_validation(oracle));
    return config;
}

memory::ptr filled(engine& eng, const layout& l, float seed) {
    auto m = eng.allocate_memory(l);
    std::vector<float> v(l.count());
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = seed + static_cast<float>(i % 13) * 0.25f - 1.5f;
    }
    if (l.data_type == data_types::f32) {
        set_values(m, v);
    }
    return m;
}

layout l4(int64_t c = 16) {
    return layout{ov::PartialShape{1, c, 8, 8}, data_types::f32, format::bfyx};
}

topology two_chains() {
    topology t;
    t.add(input_layout("in1", l4(4)));
    t.add(input_layout("in2", l4(4)));
    t.add(activation("a1", input_info("in1"), activation_func::relu));
    t.add(activation("a2", input_info("a1"), activation_func::abs));
    t.add(activation("oa", input_info("a2"), activation_func::abs));
    t.add(activation("b1", input_info("in2"), activation_func::relu));
    t.add(activation("b2", input_info("b1"), activation_func::abs));
    t.add(activation("ob", input_info("b2"), activation_func::abs));
    return t;
}

topology concat_crop() {
    topology t;
    t.add(input_layout("in1", l4()));
    t.add(input_layout("in2", l4()));
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(reorder("out", input_info("C"), format::bfyx, data_types::f32));
    t.add(input_layout("in3", l4()));
    t.add(activation("B1", input_info("in3"), activation_func::relu));
    t.add(crop("K", input_info("B1"), tensor(1, 8, 8, 8), tensor(0, 0, 0, 0)));
    t.add(activation("outK", input_info("K"), activation_func::abs));
    return t;
}

void feed(engine& eng, network& net, int c) {
    for (const auto& id : {"in1", "in2", "in3"}) {
        try {
            net.set_input_data(id, filled(eng, l4(c), 1.f));
        } catch (...) {
        }
    }
}

std::string dump(const alias_oracle_report& r) {
    std::ostringstream os;
    os << "ran=" << r.ran << " skipped=" << r.skipped_reason << " v2b reads=" << r.v2b_reads_checked << " writes=" << r.v2b_writes
       << " unverifiable=" << r.v2b_unverifiable << " excluded=" << r.v2b_excluded << " defects=" << r.v2b_defect_count() << "\n";
    for (const auto& h : r.v2b_hits) {
        os << (h.sev == severity::defect ? "  D " : "  I ") << h.detail << "\n";
    }
    return os.str();
}

// Test-only fault injection: removes the restriction (both directions) between two nodes. The set is a plain sorted vector for graphs of
// this size; the program was built by the normal pipeline, the network is created afterwards so the pool sees the weakened restrictions.
void erase_restriction(program& prog, const std::string& a, const std::string& b) {
    auto erase_one = [&](program_node& from, const program_node& to) {
        auto& v = const_cast<std::vector<uint32_t>&>(from.get_memory_dependencies());
        v.erase(std::remove(v.begin(), v.end(), static_cast<uint32_t>(to.get_unique_id())), v.end());
    };
    auto& na = prog.get_node(a);
    auto& nb = prog.get_node(b);
    erase_one(na, nb);
    erase_one(nb, na);
    ASSERT_FALSE(na.has_memory_dependency(static_cast<uint32_t>(nb.get_unique_id())));
    ASSERT_FALSE(nb.has_memory_dependency(static_cast<uint32_t>(na.get_unique_id())));
}

}  // namespace

// V2b-1. Correct graphs: nothing is reported and the check really looked at reads and writes.
TEST(alias_oracle_v2b, V2b_1_correct_graphs_have_no_defect) {
    auto& eng = get_test_engine();
    for (int variant = 0; variant < 2; ++variant) {
        auto prog = program::build_program(eng, variant == 0 ? two_chains() : concat_crop(), make_cfg(eng, true, variant != 0), false, false);
        network net(prog, 0);
        feed(eng, net, variant == 0 ? 4 : 16);
        net.execute();
        ASSERT_NE(net.get_alias_oracle_report(), nullptr);
        const auto& rep = *net.get_alias_oracle_report();
        ASSERT_TRUE(rep.ran) << rep.skipped_reason;
        EXPECT_GT(rep.v2b_reads_checked, 0u) << dump(rep);
        EXPECT_GT(rep.v2b_writes, 0u) << dump(rep);
        EXPECT_EQ(rep.v2b_defect_count(), 0u) << dump(rep);
        EXPECT_EQ(rep.defect_count(), 0u);   // V0 / V2 / V3 are unchanged by V2b
    }
}

// V2b-2. Fault injection: the restrictions between the two chains are deleted, the pool then lets interleaved lifetimes share a block and
// V2b must report the clobber. The control run (no deletion) is clean.
TEST(alias_oracle_v2b, V2b_2_fault_injection_deleted_restriction_is_reported) {
    auto& eng = get_test_engine();
    {   // control
        auto prog = program::build_program(eng, two_chains(), make_cfg(eng, true, false), false, false);
        network net(prog, 0);
        feed(eng, net, 4);
        net.execute();
        ASSERT_NE(net.get_alias_oracle_report(), nullptr);
        EXPECT_EQ(net.get_alias_oracle_report()->v2b_defect_count(), 0u) << dump(*net.get_alias_oracle_report());
    }
    auto prog = program::build_program(eng, two_chains(), make_cfg(eng, true, false), false, false);
    for (const auto* x : {"a1", "a2"}) {
        for (const auto* y : {"b1", "b2"}) {
            erase_restriction(*prog, x, y);
        }
    }
    network net(prog, 0);
    feed(eng, net, 4);
    net.execute();
    ASSERT_NE(net.get_alias_oracle_report(), nullptr);
    const auto& rep = *net.get_alias_oracle_report();
    ASSERT_TRUE(rep.ran);
    // the in-order axis interleaves the chains (BFS); if it did not, the construction would not produce a clobber
    const auto axis = exec_axis::build(*prog);
    ASSERT_TRUE(axis.valid());
    const auto pa1 = *axis.position(prog->get_node("a1")), pa2 = *axis.position(prog->get_node("a2"));
    const auto pb1 = *axis.position(prog->get_node("b1")), pb2 = *axis.position(prog->get_node("b2"));
    ASSERT_TRUE((pb1 > pa1 && pb1 < pa2) || (pb2 > pa1 && pb2 < pa2) || (pa1 > pb1 && pa1 < pb2) || (pa2 > pb1 && pa2 < pb2))
        << "chains are not interleaved on the axis";
    EXPECT_GE(rep.v2b_defect_count(), 1u) << dump(rep);
}

// V2b-3. Independence from alias_analysis and class_lifetimes: b1's output is re-bound onto the live buffer of a1 by hand. No alias edge, no
// restriction and no class knows about it, the lifetime engine calls both nodes eligible and disjoint, V2b still sees the write.
TEST(alias_oracle_v2b, V2b_3_unmodeled_alias_is_seen_without_alias_analysis) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, two_chains(), make_cfg(eng, true, false), false, false);
    network net(prog, 0);
    const auto axis = exec_axis::build(*prog);
    ASSERT_TRUE(axis.valid());
    const auto* aa = prog->get_alias_analysis();
    ASSERT_NE(aa, nullptr);
    const auto lt = class_lifetimes::build(*prog, *aa, axis);
    const auto pa1 = *axis.position(prog->get_node("a1")), pa2 = *axis.position(prog->get_node("a2"));
    std::string victim;
    for (const auto* v : {"b1", "b2"}) {
        const auto p = *axis.position(prog->get_node(v));
        if (p > pa1 && p < pa2) {
            victim = v;
        }
    }
    ASSERT_FALSE(victim.empty()) << "no chain-b node is executed between a1 and its reader a2";
    EXPECT_TRUE(lt.eligible(prog->get_node("a1")));
    EXPECT_TRUE(lt.eligible(prog->get_node(victim)));
    EXPECT_TRUE(lt.classes_disjoint(prog->get_node("a1"), prog->get_node(victim)));
    auto a1 = net.get_primitive("a1");
    auto bv = net.get_primitive(victim);
    ASSERT_NE(a1->output_memory_ptr(0), nullptr);
    bv->set_output_memory(a1->output_memory_ptr(0), false, 0);   // the unmodeled alias
    feed(eng, net, 4);
    net.execute();
    ASSERT_NE(net.get_alias_oracle_report(), nullptr);
    const auto& rep = *net.get_alias_oracle_report();
    ASSERT_TRUE(rep.ran);
    bool seen = false;
    for (const auto& h : rep.v2b_hits) {
        seen = seen || (h.sev == severity::defect && h.a == victim && h.b == "a2");
    }
    EXPECT_TRUE(seen) << dump(rep);
    // the engine's own view of the same two classes is unchanged by the hand made alias (it never looked at memory)
    EXPECT_TRUE(lt.eligible(prog->get_node("a1")));
}

// V2b-4. selective_ssm re-points output 1 to input 5 (F27, no alias edge): the oracle runs through without a spurious defect.
TEST(alias_oracle_v2b, V2b_4_selective_ssm_empty_sequence_runs_clean) {
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
    program::ptr prog;
    try {
        prog = program::build_program(eng, t, make_cfg(eng), false, false);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "selective_ssm graph cannot be built here: " << e.what();
    }
    network net(prog, 0);
    net.set_input_data("A", filled(eng, layout{ov::PartialShape{heads}, data_types::f32, format::bfyx}, 1.f));
    net.set_input_data("dt", eng.allocate_memory(layout{ov::PartialShape{1, 0, heads}, data_types::f32, format::bfyx}));
    net.set_input_data("B", eng.allocate_memory(layout{ov::PartialShape{1, 0, groups, state_size}, data_types::f32, format::bfyx}));
    net.set_input_data("x", eng.allocate_memory(layout{ov::PartialShape{1, 0, heads, head_dim}, data_types::f32, format::bfyx}));
    net.set_input_data("C", eng.allocate_memory(layout{ov::PartialShape{1, 0, groups, state_size}, data_types::f32, format::bfyx}));
    net.set_input_data("state_in", filled(eng, layout{ov::PartialShape{1, heads, head_dim, state_size}, data_types::f32, format::bfyx}, 1.f));
    net.execute();
    ASSERT_NE(net.get_alias_oracle_report(), nullptr);
    const auto& rep = *net.get_alias_oracle_report();
    EXPECT_TRUE(rep.ran) << rep.skipped_reason;
    EXPECT_EQ(rep.v2b_defect_count(), 0u) << dump(rep);
}

// V2b-5. Switch off: no oracle, no V2b (the oracle is exactly as before).
TEST(alias_oracle_v2b, V2b_5_switch_off_no_report) {
    auto& eng = get_test_engine();
    auto prog = program::build_program(eng, two_chains(), make_cfg(eng, false), false, false);
    network net(prog, 0);
    feed(eng, net, 4);
    net.execute();
    EXPECT_EQ(net.get_alias_oracle_report(), nullptr);
}
