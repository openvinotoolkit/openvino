// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression tests for F7: add_required_reorders::eliminate_pad_for_onednn_impl must not clear the spatial padding
// of a reorder that is an in-place concat predecessor (the padding positions the reorder inside the concat buffer).
//
// Topology (primary): R1 = reorder(in1), R2 = reorder(in2), C = concat({R2, R1}, axis) (in-place),
// A = abs(C) (ocl), N = gemm(R1, in3) forced to oneDNN (plain input formats, so no reorder is inserted in front of N).
// The axis-1 control only has feature padding, which eliminate_pad_for_onednn_impl keeps (early return).

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/gemm.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"

#include "program_node.h"
#include "primitive_inst.h"

#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

using namespace cldnn;
using namespace ::tests;

namespace {

std::vector<float> read_f32(memory::ptr mem) {
    mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
    return std::vector<float>(lock.begin(), lock.end());
}

std::string dump_graph(program& prog) {
    std::ostringstream os;
    for (auto* n : prog.get_processing_order()) {
        const auto& pad = n->get_output_layout().data_padding;
        os << "[GRAPH] #" << prog.get_processing_order().get_processing_number(n) << " " << n->id()
           << " type=" << n->get_primitive()->type_string() << " can_be_optimized=" << n->can_be_optimized()
           << " is_output=" << n->is_output() << " impl=" << static_cast<int>(n->get_preferred_impl_type())
           << " fmt=" << n->get_output_layout().format.to_string() << " pad_lower=[";
        for (size_t i = 0; i < 4; i++) {
            os << pad._lower_size[i] << (i < 3 ? "," : "");
        }
        os << "] pad_upper=[";
        for (size_t i = 0; i < 4; i++) {
            os << pad._upper_size[i] << (i < 3 ? "," : "");
        }
        os << "] deps={";
        for (const auto& d : n->get_dependencies()) {
            os << d.first->id() << ",";
        }
        os << "} users={";
        for (auto* u : n->get_users()) {
            os << u->id() << ",";
        }
        os << "}\n";
    }
    return os.str();
}

memory::ptr make_input(engine& eng, data_types dt, int variant) {
    layout l{ov::PartialShape{1, 16, 8, 8}, dt, format::bfyx};
    auto mem = eng.allocate_memory(l);
    const size_t n = l.count();
    if (dt == data_types::f16) {
        std::vector<ov::float16> v(n);
        for (size_t i = 0; i < n; i++) {
            v[i] = ov::float16(static_cast<float>((i * (3 + variant)) % 13) * 0.0625f - 0.5f);
        }
        set_values(mem, v);
    } else {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; i++) {
            v[i] = static_cast<float>((i * (3 + variant)) % 17) * 0.125f - 1.0f;
        }
        set_values(mem, v);
    }
    return mem;
}

struct f7_graph {
    topology t;
    ov::intel_gpu::ImplForcingMap forced;
    std::vector<std::string> outs;
};

f7_graph build_f7(int64_t axis) {
    f7_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{ov::PartialShape{1, 16, 8, 8}, data_types::f32, format::bfyx}));
    t.add(input_layout("in2", layout{ov::PartialShape{1, 16, 8, 8}, data_types::f32, format::bfyx}));
    t.add(input_layout("in3", layout{ov::PartialShape{1, 16, 8, 8}, data_types::f16, format::bfyx}));
    t.add(reorder("R1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("R2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("R2"), input_info("R1")}, axis));
    t.add(activation("A", input_info("C"), activation_func::abs));
    t.add(reorder("out_A", input_info("A"), format::bfyx, data_types::f32));
    t.add(gemm("N", {input_info("R1"), input_info("in3")}, data_types::f16, false, true, 1.0f, 0.0f));
    t.add(reorder("out_N", input_info("N"), format::bfyx, data_types::f32));
    g.outs = {"out_A", "out_N"};
    g.forced = {{"A", ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::ocl}},
                {"N", ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::onednn}}};
    return g;
}

using host_map = std::map<std::string, std::vector<float>>;

host_map run_net(network& net, const std::map<std::string, memory::ptr>& inputs, const std::vector<std::string>& out_ids) {
    for (const auto& kv : inputs) {
        net.set_input_data(kv.first, kv.second);
    }
    auto outs = net.execute();
    net.get_stream().finish();
    host_map r;
    for (const auto& id : out_ids) {
        r[id] = read_f32(outs.at(id).get_memory());
    }
    return r;
}

void expect_outputs_match(const host_map& got, const host_map& ref) {
    for (const auto& kv : ref) {
        const auto& g = got.at(kv.first);
        ASSERT_EQ(g.size(), kv.second.size()) << kv.first;
        float max_diff = 0.f;
        size_t bad = 0;
        for (size_t i = 0; i < g.size(); i++) {
            const float d = std::fabs(g[i] - kv.second[i]);
            max_diff = std::max(max_diff, d);
            if (d > 0.02f + 0.01f * std::fabs(kv.second[i])) {
                bad++;
            }
        }
        std::cout << "[CMP] " << kv.first << " max_diff=" << max_diff << " mismatches=" << bad << "/" << g.size() << std::endl;
        EXPECT_EQ(bad, 0u) << "output " << kv.first << " does not match the reference (max_diff=" << max_diff << ")";
    }
}

bool has_any_padding(const padding& p) {
    for (size_t i = 0; i < 4; i++) {
        if (p._lower_size[i] != 0 || p._upper_size[i] != 0) {
            return true;
        }
    }
    return false;
}

void run_f7(int64_t axis) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN gemm needs supports_immad";
    }
    auto g = build_f7(axis);
    std::map<std::string, memory::ptr> inputs{{"in1", make_input(engine, data_types::f32, 1)},
                                              {"in2", make_input(engine, data_types::f32, 2)},
                                              {"in3", make_input(engine, data_types::f16, 3)}};

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(g.forced));
    ASSERT_EQ(config.get_queue_type(), QueueTypes::in_order);
    auto prog = program::build_program(engine, g.t, config);
    network net(prog, 0);
    const std::string graph = dump_graph(*prog);
    std::cout << graph;

    // ---- graph-shape preconditions (stop and report if any fails; do not retune the topology).
    // Pre-fix: R1.users == {C, N} and N.dep(0) == R1. Post-fix N.dep(0) is an inserted padding reorder fed by R1.
    auto& R1 = prog->get_node("R1");
    auto& C = prog->get_node("C");
    auto& N = prog->get_node("N");
    ASSERT_TRUE(C.can_be_optimized()) << graph;
    ASSERT_TRUE(R1.is_type<reorder>()) << graph;
    ASSERT_EQ(N.get_preferred_impl_type(), impl_types::onednn) << graph;
    ASSERT_NE(N.get_selected_impl(), nullptr) << graph;
    ASSERT_FALSE(N.get_selected_impl()->is_cpu()) << graph;
    const program_node* n_in = &N.get_dependency(0);
    const bool n_reads_r1_directly = (n_in == &R1);
    if (!n_reads_r1_directly) {
        ASSERT_TRUE(n_in->is_type<reorder>()) << graph;
        ASSERT_EQ(n_in->get_dependencies().size(), 1u) << graph;
        ASSERT_EQ(&n_in->get_dependency(0), &R1) << graph;
    }
    std::set<std::string> u;
    for (auto* x : R1.get_users()) {
        u.insert(x->id());
    }
    ASSERT_TRUE(u.count("C")) << graph;
    ASSERT_TRUE(u.count(n_reads_r1_directly ? N.id() : n_in->id())) << graph;
    ASSERT_EQ(u.size(), 2u) << graph;

    auto res = run_net(net, inputs, g.outs);

    auto gref = build_f7(axis);
    ExecutionConfig ref_cfg = get_test_default_config(engine);
    ref_cfg.set_property(ov::intel_gpu::optimize_data(false));
    network net_ref(engine, gref.t, ref_cfg);
    auto ref = run_net(net_ref, inputs, gref.outs);

    const auto& r1_pad = R1.get_output_layout().data_padding;
    std::cout << "[F7] axis=" << axis << " N_reads_R1_directly=" << n_reads_r1_directly << " R1 lower[1]=" << r1_pad._lower_size[1]
              << " lower[2]=" << r1_pad._lower_size[2] << std::endl;
    if (axis == 2) {
        // R1 is the second concat input: it must keep its slot offset, N gets an unpadded copy.
        EXPECT_EQ(r1_pad._lower_size[2], 8) << "R1 lost its concat slot padding\n" << graph;
        EXPECT_FALSE(n_reads_r1_directly) << "N must read an unpadded copy of R1\n" << graph;
        if (!n_reads_r1_directly) {
            EXPECT_FALSE(has_any_padding(n_in->get_output_layout().data_padding)) << graph;
        }
    } else {
        EXPECT_TRUE(n_reads_r1_directly) << graph;
        EXPECT_EQ(r1_pad._lower_size[1], 16) << graph;
    }
    expect_outputs_match(res, ref);
}

}  // namespace

// Spatial in-place concat predecessor consumed by a oneDNN gemm: padding must not be cleared.
TEST(inplace_alias_conflict_f7, F7_spatial_concat_reorder_pred_with_onednn_gemm_user) {
    run_f7(2);
}

// Control: feature-only padding is kept by eliminate_pad_for_onednn_impl (early return), before and after the fix.
TEST(inplace_alias_conflict_f7, F7_control_feature_axis_concat) {
    run_f7(1);
}
