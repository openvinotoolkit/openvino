// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// P2-a L31 (audit C11): cross-binary neutrality corpus. DISABLED by default (it is a data producer, not a check). Run it with
//     OV_GPU_L31_DUMP_DIR=<dir> ov_gpu_unit_tests --gtest_also_run_disabled_tests --gtest_filter='l31_dump.DISABLED_corpus'
// in BOTH the binary of the commit under test and a binary built at the baseline (the same file is dropped into the baseline tree; it only
// uses APIs that exist at the baseline and never touches enable_inorder_lifetime_restrictions, i.e. it exercises "switch off" on both sides).
// For each (topology, queue) it writes <dir>/<tag>.memdeps.txt (same text as program::get_memory_dependencies_string()), <tag>.trace.txt
// (allocation trace records, debug caps builds) and <tag>.out.txt (output values). tests/unit/passes/l31_compare.sh diffs two directories.

#include "test_utils.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "program_node.h"

using namespace cldnn;
using namespace ::tests;

namespace {

layout l4(int64_t c = 16, data_types dt = data_types::f32) {
    return layout{ov::PartialShape{1, c, 8, 8}, dt, format::bfyx};
}

topology pattern_a() {
    topology t;
    t.add(input_layout("in", l4()));
    t.add(activation("P", input_info("in"), activation_func::relu));
    t.add(crop("K", input_info("P"), tensor(1, 8, 8, 8), tensor(0, 0, 0, 0)));
    for (int k = 0; k < 3; ++k) {
        t.add(activation("X" + std::to_string(k), input_info("P"), activation_func::relu));
        t.add(activation("Y" + std::to_string(k), input_info("X" + std::to_string(k)), activation_func::abs));
    }
    t.add(eltwise("J1", {input_info("Y0"), input_info("Y1")}, eltwise_mode::sum));
    t.add(eltwise("J", {input_info("J1"), input_info("Y2")}, eltwise_mode::sum));
    t.add(activation("R", input_info("K"), activation_func::abs));
    t.add(reorder("O", input_info("J"), format::bfyx, data_types::f32));
    t.add(activation("OR", input_info("R"), activation_func::abs));
    return t;
}

topology concat_inplace() {
    topology t;
    t.add(input_layout("in1", l4()));
    t.add(input_layout("in2", l4()));
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(activation("Q", input_info("C"), activation_func::abs));
    t.add(reorder("out", input_info("Q"), format::bfyx, data_types::f32));
    t.add(input_layout("in3", l4()));
    t.add(activation("B1", input_info("in3"), activation_func::relu));
    t.add(crop("K", input_info("B1"), tensor(1, 8, 8, 8), tensor(0, 0, 0, 0)));
    t.add(activation("outK", input_info("K"), activation_func::abs));
    return t;
}

topology residual_chain() {
    topology t;
    t.add(input_layout("in", l4()));
    t.add(activation("a0", input_info("in"), activation_func::relu));
    t.add(activation("a1", input_info("a0"), activation_func::abs));
    t.add(eltwise("s1", {input_info("a1"), input_info("a0")}, eltwise_mode::sum));
    t.add(activation("a2", input_info("s1"), activation_func::relu));
    t.add(eltwise("s2", {input_info("a2"), input_info("s1")}, eltwise_mode::sum));
    t.add(reorder("out", input_info("s2"), format::bfyx, data_types::f32));
    return t;
}

topology dynamic_branches() {
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

std::vector<float> read_f32(const network_output& o) {
    auto m = o.get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> p(m, get_test_stream());
    return std::vector<float>(p.begin(), p.begin() + std::min<size_t>(p.size(), o.get_layout().count()));
}

void write(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary);
    f << text;
}

void dump_one(engine& eng, const std::string& dir, const std::string& tag, const topology& topo, QueueTypes q, const std::vector<std::string>& outputs,
              bool dynamic) {
    ExecutionConfig cfg = get_test_default_config(eng);
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cfg.set_property(ov::intel_gpu::queue_type(q));
    if (q == QueueTypes::out_of_order) {
        cfg.set_user_property({ov::intel_gpu::use_onednn(false)}, ov::OptionVisibility::RELEASE_INTERNAL);
    }
    cfg.set_user_property({ov::intel_gpu::dump_alloc_trace(2)});
    auto prog = program::build_program(eng, topo, cfg, false, false);
    ASSERT_EQ(prog->get_config().get_queue_type(), q) << tag;
    std::ostringstream md;
    for (auto* n : prog->get_processing_order()) {
        md << "primitive: " << n->id() << "(unique_id:" << n->get_unique_id() << ") restricted list: ";
        for (auto uid : n->get_memory_dependencies()) {
            md << uid << ",";
        }
        md << "\n";
    }
    write(dir + "/" + tag + ".memdeps.txt", md.str());

    network net(prog, 0);
    std::ostringstream out;
    for (int round = 0; round < (dynamic ? 3 : 1); ++round) {
        const int64_t c = dynamic ? 4 + 4 * round : 16;
        for (const auto& id : {"in", "in1", "in2", "in3"}) {
            try {
                net.set_input_data(id, filled(eng, dynamic ? layout{ov::PartialShape{1, c, 8, 8}, data_types::f32, format::bfyx} : l4(), 1.f + round));
            } catch (...) {
            }
        }
        auto outs = net.execute();
        for (const auto& id : outputs) {
            out << id << " round " << round << ":";
            for (float x : read_f32(outs.at(id))) {
                out << " " << x;
            }
            out << "\n";
        }
    }
    write(dir + "/" + tag + ".out.txt", out.str());
#ifdef GPU_DEBUG_CONFIG
    std::ostringstream tr;
    for (const auto& rec : net.get_alloc_trace_records()) {
        tr << rec.node_id << ":" << rec.port << " " << rec.bytes << " pooled=" << rec.pooled << " " << rec.reason << "\n";
    }
    write(dir + "/" + tag + ".trace.txt", tr.str());
#endif
}

}  // namespace

TEST(l31_dump, DISABLED_corpus) {
    const char* dir = std::getenv("OV_GPU_L31_DUMP_DIR");
    ASSERT_NE(dir, nullptr) << "set OV_GPU_L31_DUMP_DIR";
    auto& eng = get_test_engine();
    // the optimizer rewrites the primitives of a topology in place: every run gets a fresh topology
    struct item {
        const char* name;
        topology (*make)();
        std::vector<std::string> outs;
        bool dyn;
    };
    const std::vector<item> corpus = {
        {"pattern_a", pattern_a, {"O", "OR"}, false},
        {"concat_inplace", concat_inplace, {"out", "outK"}, false},
        {"residual", residual_chain, {"out"}, false},
        {"dynamic", dynamic_branches, {"output"}, true},
    };
    for (const auto& it : corpus) {
        dump_one(eng, dir, std::string(it.name) + ".in_order", it.make(), QueueTypes::in_order, it.outs, it.dyn);
        dump_one(eng, dir, std::string(it.name) + ".ooo", it.make(), QueueTypes::out_of_order, it.outs, it.dyn);
    }
}
