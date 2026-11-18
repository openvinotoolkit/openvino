// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Pre-0 F19: a static oneDNN sum writer W whose outer dependency is an optimized (view) node bound its dst at construction
// to the view's private buffer, and was never re-pointed when the view was re-aliased to its owner in
// network::allocate_primitives(). Fix: primitive_inst::rebind_reused_eltw_output(), sum only (binary_mul_inplace keeps its
// private dst, see design section 9 R1).
//
// Every sum test asserts, after network construction and BEFORE the first execute():
//   same_buffer(W.out, dep.out) == 1
// and compares the outputs with a non-optimized reference elementwise.

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"

#include "pass_manager.h"
#include "program_helpers.h"
#include "program_node.h"
#include "primitive_inst.h"

#include <cmath>
#include <iostream>
#include <map>
#include <sstream>

using namespace cldnn;
using namespace ::tests;

namespace {

using fusing = onednn_eltwise_fusing_helpers;

const ov::PartialShape kShape{32, 16};

std::vector<float> to_f32(memory::ptr mem) {
    std::vector<float> r;
    if (mem->get_layout().data_type == data_types::f16) {
        mem_lock<ov::float16, mem_lock_type::read> lock(mem, get_test_stream());
        for (auto v : lock) {
            r.push_back(static_cast<float>(v));
        }
    } else {
        mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
        r.assign(lock.begin(), lock.end());
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

ExecutionConfig opt_config(engine& eng, const ov::intel_gpu::ImplForcingMap& forced, const std::vector<std::string>& custom_outputs,
                           QueueTypes q = QueueTypes::in_order) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(forced));
    config.set_property(ov::intel_gpu::queue_type(q));
    if (!custom_outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(custom_outputs));
    }
    return config;
}

ExecutionConfig ref_config(engine& eng, const std::vector<std::string>& custom_outputs) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    if (!custom_outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(custom_outputs));
    }
    return config;
}

const fused_primitive_desc* fused_eltw(const program_node& p, eltwise_mode mode) {
    for (const auto& fd : p.get_fused_primitives()) {
        if (fd.is_type<eltwise>() && fd.typed_desc<eltwise>()->mode == mode && fd.deps.size() == 1 && fd.has_outer_dep()) {
            return &fd;
        }
    }
    return nullptr;
}

std::string dump_flags(program& prog) {
    std::ostringstream os;
    for (auto* n : prog.get_processing_order()) {
        os << "[FLAGS] #" << prog.get_processing_order().get_processing_number(n) << " " << n->id()
           << " type=" << n->get_primitive()->type_string() << " can_be_optimized=" << n->can_be_optimized()
           << " runtime_skippable=" << n->is_runtime_skippable() << " is_output=" << n->is_output()
           << " fused=" << n->get_fused_primitives().size() << " users=" << n->get_users().size()
           << " can_share_buffer=" << n->can_share_buffer() << " impl=" << static_cast<int>(n->get_preferred_impl_type())
           << " fmt=" << n->get_output_layout().format.to_string() << "\n";
    }
    return os.str();
}

struct f19_graph {
    topology t;
    ov::intel_gpu::ImplForcingMap forced;
    std::vector<std::string> outs;
    std::vector<std::string> custom_outputs;
    std::vector<std::pair<std::string, ov::PartialShape>> inputs;
};

std::map<std::string, std::vector<float>> run_net(network& net, engine& eng, const f19_graph& g) {
    int variant = 1;
    for (const auto& in : g.inputs) {
        net.set_input_data(in.first, make_f16(eng, in.second, variant++));
    }
    auto outs = net.execute();
    std::map<std::string, std::vector<float>> r;
    for (const auto& id : g.outs) {
        r[id] = to_f32(outs.at(id).get_memory());
    }
    return r;
}

void expect_match(const std::map<std::string, std::vector<float>>& got, const std::map<std::string, std::vector<float>>& ref,
                  const std::string& tag) {
    for (const auto& kv : ref) {
        const auto& g = got.at(kv.first);
        ASSERT_EQ(g.size(), kv.second.size()) << kv.first;
        size_t bad = 0;
        float max_diff = 0.f;
        for (size_t i = 0; i < g.size(); i++) {
            const float d = std::fabs(g[i] - kv.second[i]);
            max_diff = std::max(max_diff, d);
            if (d > 0.02f + 0.01f * std::fabs(kv.second[i])) {
                bad++;
            }
        }
        std::cout << "[CMP " << tag << "] " << kv.first << " max_diff=" << max_diff << " mismatches=" << bad << "/" << g.size() << std::endl;
        EXPECT_EQ(bad, 0u) << tag << ": output " << kv.first << " does not match the reference";
    }
}

struct sum_case_opts {
    QueueTypes queue = QueueTypes::in_order;
    bool legacy_shape_infer = false;  // do not force allow_new_shape_infer (conv/resample topologies)
    bool expect_same_buffer = true;  // sum cases: W.out must alias dep.out right after construction
};

// W (oneDNN) has a fused sum whose outer dep is `dep_id`. Returns via ASSERT_*.
void run_sum_case(const f19_graph& g, const std::string& tag, const std::string& w_id, const std::string& dep_id,
                  const sum_case_opts& o = {}) {
    auto& engine = get_test_engine();
    auto cfg = opt_config(engine, g.forced, g.custom_outputs, o.queue);
    if (o.legacy_shape_infer) {
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(false));
    }
    auto prog = program::build_program(engine, g.t, cfg);
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << "[" << tag << "]\n" << flags;
    auto& W = prog->get_node(w_id);
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    const auto* desc = fused_eltw(W, eltwise_mode::sum);
    ASSERT_NE(desc, nullptr) << "W has no fused eltwise sum\n" << flags;
    ASSERT_EQ(W.get_dependency(desc->outer_dep_start_idx).id(), dep_id) << flags;
    ASSERT_TRUE(fusing::get_add_fusing_type(W, *desc) == add_fusing_type::sum) << flags;
    ASSERT_NE(fusing::get_reused_eltwmem_idx(W), -1) << flags;

    auto w_inst = net.get_primitive(w_id);
    auto dep_inst = net.get_primitive(dep_id);
    ASSERT_GE(w_inst->get_reused_eltw_dep_idx(), 0);
    const bool same = engine.is_the_same_buffer(*w_inst->output_memory_ptr(), *dep_inst->output_memory_ptr());
    std::cout << "[" << tag << "] same_buffer(W.out, dep.out) before execute = " << same << std::endl;
    EXPECT_EQ(same, o.expect_same_buffer) << "I-F19-1 violated: sum dst is not bound to the dependency buffer after construction";

    auto res = run_net(net, engine, g);
    auto rcfg = ref_config(engine, g.custom_outputs);
    if (o.legacy_shape_infer) {
        rcfg.set_property(ov::intel_gpu::allow_new_shape_infer(false));
    }
    network net_ref(engine, g.t, rcfg);
    auto ref = run_net(net_ref, engine, g);
    expect_match(res, ref, tag);
}

#define SKIP_IF_NO_IMMAD()                                                                \
    do {                                                                                  \
        if (!get_test_engine().get_device_info().supports_immad) {                        \
            GTEST_SKIP() << "oneDNN sum needs supports_immad";                            \
        }                                                                                 \
    } while (0)

// O = FC(in3) (ocl); V = no-op strided_slice(O); W = FC(in1) (onednn); sum(W, V); Z = FC(S) ; out_Z.
f19_graph build_base(engine& eng) {
    f19_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", kShape}, {"in3", kShape}};
    t.add(data("wO", make_weights(eng, 16, 16, 1)));
    t.add(fully_connected("O", input_info("in3"), "wO"));
    t.add(strided_slice("V", input_info("O"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
    t.add(data("wW", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("S", {input_info("W"), input_info("V")}, eltwise_mode::sum));
    t.add(data("wZ", make_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("S"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("O", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    g.outs.push_back("out_Z");
    return g;
}

}  // namespace

// 2. resample (optimized-out) feeds the oneDNN sum. Same topology as
// basic_memory_dependencies.optimized_resample_to_onednn_sum_reuse_correctness, with the construction-time assertion.
TEST(inplace_alias_conflict_f19, F19_resample_view_bound_at_construction) {
    SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    auto in_layout = layout{ov::PartialShape({1, 16, 32, 32}), data_types::f16, format::bfyx};
    auto weight_layout = layout{ov::PartialShape({16, 16, 1, 1}), data_types::f16, format::bfyx};
    auto weight_mem = engine.allocate_memory(weight_layout);
    {
        std::vector<ov::float16> v(weight_layout.count());
        for (size_t i = 0; i < v.size(); i++) {
            v[i] = ov::float16(static_cast<float>((i * 7) % 11) * 0.02f - 0.1f);
        }
        set_values(weight_mem, v);
    }
    f19_graph g;
    auto& t = g.t;
    t.add(input_layout("input", in_layout));
    g.inputs = {{"input", in_layout.get_partial_shape()}};
    t.add(data("weight", weight_mem));
    t.add(convolution("conv1", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    t.add(convolution("conv2", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    t.add(convolution("conv2b", input_info("conv2"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    t.add(convolution("conv3", input_info("conv2b"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    t.add(resample("resample", input_info("conv1"), tensor(1, 16, 32, 32), 1, ov::op::v4::Interpolate::InterpolateMode::NEAREST));
    t.add(eltwise("eltwise", input_info("conv3"), input_info("resample"), eltwise_mode::sum));
    t.add(reorder("reorder", input_info("eltwise"), format::bfyx, data_types::f32));
    g.outs = {"reorder"};
    const auto fsv16 = format::b_fs_yx_fsv16;
    g.forced = ov::intel_gpu::ImplForcingMap{{"conv1", {fsv16, "", impl_types::onednn}},
                                             {"conv2", {fsv16, "", impl_types::onednn}},
                                             {"conv2b", {fsv16, "", impl_types::onednn}},
                                             {"resample", {fsv16, "", impl_types::ocl}},
                                             {"conv3", {fsv16, "", impl_types::onednn}},
                                             {"eltwise", {fsv16, "", impl_types::ocl}}};
    sum_case_opts o;
    o.legacy_shape_infer = true;
    run_sum_case(g, "F19_resample", "conv3", "resample", o);
}

// 1b. strided_slice view of an internal producer (the F6h topology) in an independent form, plus the can_share_buffer guard
// on the sum root.
TEST(inplace_alias_conflict_f19, F19_strided_slice_view_of_internal_producer) {
    SKIP_IF_NO_IMMAD();
    auto g = build_base(get_test_engine());
    run_sum_case(g, "F19_ss_view", "W", "V");
}

// 3. view chain O -> RS1 (reshape to 3D) -> SS (no-op strided_slice) -> RS2 (reshape back to 2D) -> W (2D FC).
TEST(inplace_alias_conflict_f19, F19_two_view_chain_reshape_of_strided_slice) {
    SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    f19_graph g;
    auto& t = g.t;
    const ov::PartialShape shape3{2, 16, 16};
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", kShape}, {"in3", kShape}};
    t.add(data("wO", make_weights(eng, 16, 16, 1)));
    t.add(fully_connected("O", input_info("in3"), "wO"));
    t.add(reshape("RS1", input_info("O"), false, std::vector<int64_t>{2, 16, 16}, shape3));
    t.add(strided_slice("SS", input_info("RS1"), {0, 0, 0}, {2, 16, 16}, {1, 1, 1}, {}, {1, 1, 1}, {}, {}, {}, ov::Shape{2, 16, 16}));
    t.add(reshape("RS2", input_info("SS"), false, std::vector<int64_t>{32, 16}, kShape));
    t.add(data("wW", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("S", {input_info("W"), input_info("RS2")}, eltwise_mode::sum));
    t.add(data("wZ", make_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("S"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("O", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    g.outs = {"out_Z"};
    run_sum_case(g, "F19_two_view_chain", "W", "RS2");
}

// 4. Two writers through views, the result depends on the PO order of the rebinds:
//    O -> V0 -> W2 (sum dep V0);  W2 -> V1 -> W1 (sum dep V1).
TEST(inplace_alias_conflict_f19, F19_two_writers_through_views_po_order) {
    SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    f19_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in4", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", kShape}, {"in3", kShape}, {"in4", kShape}};
    t.add(data("wO", make_weights(eng, 16, 16, 1)));
    t.add(fully_connected("O", input_info("in3"), "wO"));
    t.add(strided_slice("V0", input_info("O"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
    t.add(data("wW2", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W2", input_info("in1"), "wW2"));
    t.add(eltwise("S2", {input_info("W2"), input_info("V0")}, eltwise_mode::sum));
    t.add(strided_slice("V1", input_info("S2"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
    t.add(data("wW1", make_weights(eng, 16, 16, 3)));
    t.add(fully_connected("W1", input_info("in4"), "wW1"));
    t.add(eltwise("S1", {input_info("W1"), input_info("V1")}, eltwise_mode::sum));
    t.add(data("wZ", make_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("S1"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("O", impl_desc(impl_types::ocl));
    g.forced.emplace("W2", impl_desc(impl_types::onednn));
    g.forced.emplace("W1", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    g.outs = {"out_Z"};

    auto& engine = get_test_engine();
    auto prog = program::build_program(engine, g.t, opt_config(engine, g.forced, g.custom_outputs));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << flags;
    auto& W1 = prog->get_node("W1");
    auto& W2 = prog->get_node("W2");
    ASSERT_NE(fused_eltw(W1, eltwise_mode::sum), nullptr) << "W1 has no fused sum\n" << flags;
    ASSERT_NE(fused_eltw(W2, eltwise_mode::sum), nullptr) << "W2 has no fused sum\n" << flags;
    ASSERT_NE(fusing::get_reused_eltwmem_idx(W1), -1) << flags;
    ASSERT_NE(fusing::get_reused_eltwmem_idx(W2), -1) << flags;
    auto w1 = net.get_primitive("W1");
    auto w2 = net.get_primitive("W2");
    auto o = net.get_primitive("O");
    EXPECT_TRUE(engine.is_the_same_buffer(*w2->output_memory_ptr(), *o->output_memory_ptr())) << "W2 dst is not O's buffer";
    EXPECT_TRUE(engine.is_the_same_buffer(*w1->output_memory_ptr(), *w2->output_memory_ptr())) << "W1 dst is not W2's buffer";
    auto res = run_net(net, engine, g);
    network net_ref(engine, g.t, ref_config(engine, g.custom_outputs));
    expect_match(res, run_net(net_ref, engine, g), "F19_two_writers");
}

// 5. Guard: binary_mul_inplace through a view of O, O has another reader after W. F19 must NOT bind W's dst to O. After F17 the
// in-place mul is rejected altogether (dep V is a view, M1): the mul must be fused and get_reused_eltwmem_idx must be -1. This test never skips on "mul did not fuse": that is a failure of the topology.
TEST(inplace_alias_conflict_f19, F19_mul_inplace_into_view_guard) {
    SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    f19_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", kShape}, {"in3", kShape}};
    t.add(data("wO", make_weights(eng, 16, 16, 1)));
    t.add(fully_connected("O", input_info("in3"), "wO"));
    t.add(strided_slice("V", input_info("O"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
    t.add(data("wW", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("P", {input_info("W"), input_info("V")}, eltwise_mode::prod));
    t.add(data("wZ", make_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("P"), "wZ"));
    // second reader of O, executed after W
    t.add(data("wQ", make_weights(eng, 16, 16, 4)));
    t.add(fully_connected("Q", input_info("O"), "wQ"));
    t.add(eltwise("R", {input_info("Z"), input_info("Q")}, eltwise_mode::sum));
    t.add(reorder("out_R", input_info("R"), format::bfyx, data_types::f32));
    g.forced.emplace("O", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    g.forced.emplace("Q", impl_desc(impl_types::ocl));
    g.outs = {"out_R"};

    auto& engine = get_test_engine();
    auto prog = program::build_program(engine, g.t, opt_config(engine, g.forced, g.custom_outputs));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << flags;
    auto& W = prog->get_node("W");
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    ASSERT_NE(fused_eltw(W, eltwise_mode::prod), nullptr) << "the prod is expected to be fused into W\n" << flags;
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W), -1) << "F17: a view dependency must not be aliased by binary_mul_inplace\n" << flags;
    // Not asserted: the frozen attrs. A no-op strided_slice becomes a view only AFTER the attribute freeze, so the freeze may still carry
    // binary_mul_inplace (the harmless direction: dst stays private). The binding points reject it, which is what is checked below.
    auto w = net.get_primitive("W");
    auto o = net.get_primitive("O");
    EXPECT_EQ(w->get_reused_eltw_dep_idx(), -1) << "F19 must only record sum writers";
    EXPECT_FALSE(engine.is_the_same_buffer(*w->output_memory_ptr(), *o->output_memory_ptr()))
        << "mul writer dst must not be bound to O (O has another reader after W)";
    // C4: no node of this graph other than a sum writer may carry a recorded reused dependency
    for (auto* n : prog->get_processing_order()) {
        if (n->is_type<data>() || n->is_type<input_layout>()) {
            continue;
        }
        EXPECT_EQ(net.get_primitive(n->id())->get_reused_eltw_dep_idx(), -1) << n->id();
    }
    auto res = run_net(net, engine, g);
    network net_ref(engine, g.t, ref_config(engine, g.custom_outputs));
    expect_match(res, run_net(net_ref, engine, g), "F19_mul_guard");
}

// 6. crop as the sum dependency. An in-place crop carries padding in its layout, which collides with the equal-padding
// requirement of the sum, so it is expected to be non-sum (unreachable). If it IS sum, the full sum checks apply.
TEST(inplace_alias_conflict_f19, F19_crop_view_dep) {
    SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    f19_graph g;
    auto& t = g.t;
    const ov::PartialShape big{64, 16};
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{big, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", kShape}, {"in3", big}};
    t.add(data("wO", make_weights(eng, 16, 16, 1)));
    t.add(fully_connected("O", input_info("in3"), "wO"));
    t.add(crop("CR", input_info("O"), tensor(32, 16, 1, 1), tensor(0, 0, 0, 0)));
    t.add(data("wW", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("S", {input_info("W"), input_info("CR")}, eltwise_mode::sum));
    t.add(data("wZ", make_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("S"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("O", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    g.outs = {"out_Z"};

    auto& engine = get_test_engine();
    auto prog = program::build_program(engine, g.t, opt_config(engine, g.forced, g.custom_outputs));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << flags;
    auto& W = prog->get_node("W");
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    const bool is_sum = fused_eltw(W, eltwise_mode::sum) != nullptr && fusing::get_reused_eltwmem_idx(W) != -1;
    auto w = net.get_primitive("W");
    if (!is_sum) {
        std::cout << "[F19_crop_view_dep] crop dep is not sum-eligible (unreachable for F19)\n";
        EXPECT_EQ(w->get_reused_eltw_dep_idx(), -1);
    } else {
        EXPECT_TRUE(engine.is_the_same_buffer(*w->output_memory_ptr(), *net.get_primitive("CR")->output_memory_ptr()));
    }
    auto res = run_net(net, engine, g);
    network net_ref(engine, g.t, ref_config(engine, g.custom_outputs));
    expect_match(res, run_net(net_ref, engine, g), "F19_crop");
}

// 7. Case 1 topology on an out-of-order queue.
TEST(inplace_alias_conflict_f19, F19_oooq_variant) {
    SKIP_IF_NO_IMMAD();
    auto g = build_base(get_test_engine());
    sum_case_opts o;
    o.queue = QueueTypes::out_of_order;
    run_sum_case(g, "F19_oooq", "W", "V", o);
}

// 8. F19-b: a static view whose owner is a dynamic node. At construction the owner has no memory (dep_memory_ptr == nullptr),
// so the writer cannot be rebound; after the owner's runtime allocation the view is re-pointed and W's dst is stale.
// Try to reproduce: O is a dynamic FC (dynamic batch input), V = static-shaped no-op strided_slice of it is impossible
// (V's shape derives from O). A static view of a dynamic owner needs a shape-agnostic view whose output shape is static
// although the input is dynamic; the shape inference gives V a dynamic shape too, and a dynamic V makes the sum
// not_supported. The test builds this graph and asserts that W is NOT a sum (the unreachable-case check).
TEST(inplace_alias_conflict_f19, F19b_static_view_of_dynamic_owner) {
    SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    f19_graph g;
    auto& t = g.t;
    const ov::PartialShape dyn{ov::Dimension::dynamic(), 16};
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{dyn, data_types::f16, format::bfyx}));
    t.add(data("wO", make_weights(eng, 16, 16, 1)));
    t.add(fully_connected("O", input_info("in3"), "wO"));
    t.add(strided_slice("V", input_info("O"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
    t.add(data("wW", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("S", {input_info("W"), input_info("V")}, eltwise_mode::sum));
    t.add(data("wZ", make_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("S"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("O", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    auto prog = program::build_program(eng, g.t, opt_config(eng, g.forced, g.custom_outputs));
    const std::string flags = dump_flags(*prog);
    std::cout << flags;
    auto& W = prog->get_node("W");
    const auto* desc = fused_eltw(W, eltwise_mode::sum);
    const bool v_dynamic = prog->get_node("V").is_dynamic();
    const bool is_sum = desc != nullptr && fusing::get_reused_eltwmem_idx(W) != -1;
    std::cout << "[F19b] V.is_dynamic=" << v_dynamic << " W_is_sum=" << is_sum << "\n";
    // Not reproducible when V is dynamic (the sum is rejected for a dynamic dependency).
    EXPECT_TRUE(v_dynamic);
    EXPECT_FALSE(is_sum) << "static sum with a dynamic-owner view: F19-b is reproducible, mark DISABLED and file separately\n" << flags;
}

// 9. The sum dependency is a view whose owner is an in-place concat holder.
TEST(inplace_alias_conflict_f19, F19_view_of_inplace_concat_holder) {
    SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    f19_graph g;
    auto& t = g.t;
    const ov::PartialShape half{16, 16};
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{half, data_types::f16, format::bfyx}));
    t.add(input_layout("in4", layout{half, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", kShape}, {"in3", half}, {"in4", half}};
    t.add(activation("P1", input_info("in3"), activation_func::abs));
    t.add(activation("P2", input_info("in4"), activation_func::relu));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 0));
    t.add(strided_slice("V", input_info("C"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
    t.add(data("wW", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("S", {input_info("W"), input_info("V")}, eltwise_mode::sum));
    t.add(data("wZ", make_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("S"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("P1", impl_desc(impl_types::ocl));
    g.forced.emplace("P2", impl_desc(impl_types::ocl));
    g.forced.emplace("C", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    g.outs = {"out_Z"};

    auto& engine = get_test_engine();
    auto prog = program::build_program(engine, g.t, opt_config(engine, g.forced, g.custom_outputs));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << flags;
    auto& W = prog->get_node("W");
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    const bool is_sum = fused_eltw(W, eltwise_mode::sum) != nullptr && fusing::get_reused_eltwmem_idx(W) != -1;
    ASSERT_TRUE(prog->get_node("C").can_be_optimized()) << "concat is not an in-place holder: the test would be vacuous\n" << flags;
    std::cout << "[F19_concat_holder] W_is_sum=" << is_sum << " C.can_be_optimized=" << prog->get_node("C").can_be_optimized()
              << " P1.can_share_buffer=" << prog->get_node("P1").can_share_buffer() << std::endl;
    if (is_sum) {
        EXPECT_FALSE(W.can_share_buffer());
        auto w = net.get_primitive("W");
        EXPECT_TRUE(engine.is_the_same_buffer(*w->output_memory_ptr(), *net.get_primitive("V")->output_memory_ptr()));
    }
    ASSERT_TRUE(is_sum) << flags;
    auto res = run_net(net, engine, g);
    network net_ref(engine, g.t, ref_config(engine, g.custom_outputs));
    expect_match(res, run_net(net_ref, engine, g), "F19_concat_holder");
}

// Sum-root guard on the base topology: the root O of the sum chain must stay out of the pool.
TEST(inplace_alias_conflict_f19, F19_sum_root_not_shareable) {
    SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    auto g = build_base(engine);
    auto prog = program::build_program(engine, g.t, opt_config(engine, g.forced, g.custom_outputs));
    EXPECT_FALSE(prog->get_node("O").can_share_buffer());
    EXPECT_FALSE(prog->get_node("W").can_share_buffer());
}
