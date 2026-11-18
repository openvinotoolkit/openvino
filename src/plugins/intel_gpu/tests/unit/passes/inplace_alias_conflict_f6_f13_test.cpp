// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression tests for
//   F6:  a oneDNN sum must not write into a buffer the network does not own (constant, mutable_data, variable, a view of a
//        graph input) or into a user-visible one (the sum dependency is a network output);
//   F13: a oneDNN sum into a graph input_layout. input_layout_inst::set_data() swaps the input's memory but the writer's
//        dst stays bound to the placeholder allocated at network build (primitive_inst.cpp ~2886-2898), so the writer
//        reads an uninitialised / stale buffer instead of the real input. Eager inputs (scalar or blocked layouts,
//        input_layout.cpp:15-25) have the identical defect.
// Plus a follow-up of F10/F11: a cascaded in-place concat slot.
//
// Every oneDNN test asserts the CORRECT behavior and the consistency chain
//   frozen_sum(W) == (get_add_fusing_type(W, desc, exact) == sum) == (get_reused_eltwmem_idx(W) != -1)
//                 == is_the_same_buffer(W.out, dep.out)           (a null buffer means "not the same buffer")
// and guard against vacuous passes: the pre-fix predicate (guard_superset) must say "sum" for the same node.
//
// F6b (sum dependency is a view of a graph input) may crash before the fix (null dst, primitive_inst.cpp ~2896); run it in
// its own process.
//
// The helper-level tests that need no oneDNN device live in inplace_alias_conflict_f6_f13_predicate_test.cpp. The predicate
// tests in this file that do not execute the network (A3, strided_slice view, A4) still need supports_immad: a sum with an
// outer dependency is only fused when oneDNN impls are enabled (prepare_primitive_fusing).

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/condition.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"

#include "pass_manager.h"
#include "program_wrapper.h"
#include "program_node.h"
#include "program_helpers.h"
#include "primitive_inst.h"

#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

using namespace cldnn;
using namespace ::tests;

namespace {

using fusing = onednn_eltwise_fusing_helpers;

std::vector<float> read_f32(memory::ptr mem) {
    mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
    return std::vector<float>(lock.begin(), lock.end());
}

std::vector<float> read_f16_as_f32(memory::ptr mem) {
    mem_lock<ov::float16, mem_lock_type::read> lock(mem, get_test_stream());
    std::vector<float> r;
    r.reserve(lock.size());
    for (auto v : lock) {
        r.push_back(static_cast<float>(v));
    }
    return r;
}

std::string dump_flags(program& prog) {
    std::ostringstream os;
    for (auto* n : prog.get_processing_order()) {
        os << "[FLAGS] #" << prog.get_processing_order().get_processing_number(n) << " " << n->id()
           << " type=" << n->get_primitive()->type_string() << " can_be_optimized=" << n->can_be_optimized()
           << " runtime_skippable=" << n->is_runtime_skippable() << " is_output=" << n->is_output()
           << " fused=" << n->get_fused_primitives().size() << " users=" << n->get_users().size()
           << " impl=" << static_cast<int>(n->get_preferred_impl_type())
           << " fmt=" << n->get_output_layout().format.to_string() << "\n";
    }
    return os.str();
}

bool frozen_sum(const program_node& p) {
    for (const auto& d : p.get_fused_primitives_onednn()) {
        if (d.op_type == onednn_post_op_type::sum) {
            return true;
        }
    }
    return false;
}

const fused_primitive_desc* fused_sum(const program_node& p) {
    for (const auto& fd : p.get_fused_primitives()) {
        if (fd.is_type<eltwise>() && fd.typed_desc<eltwise>()->mode == eltwise_mode::sum && fd.deps.size() == 1 && fd.has_outer_dep()) {
            return &fd;
        }
    }
    return nullptr;
}

// A null buffer means "not the same buffer".
bool same_buffer(engine& eng, const memory::ptr& a, const memory::ptr& b) {
    if (!a || !b) {
        return false;
    }
    return eng.is_the_same_buffer(*a, *b);
}

const ov::PartialShape kShape{32, 16};

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

ExecutionConfig opt_config(engine& eng, const ov::intel_gpu::ImplForcingMap& forced, const std::vector<std::string>& custom_outputs = {}) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(forced));
    if (!custom_outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(custom_outputs));
    }
    return config;
}

ExecutionConfig ref_config(engine& eng, const std::vector<std::string>& custom_outputs = {}) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    if (!custom_outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(custom_outputs));
    }
    return config;
}

using host_map = std::map<std::string, std::vector<float>>;

host_map run_net(network& net, const std::map<std::string, memory::ptr>& inputs, const std::vector<std::string>& out_ids) {
    for (const auto& kv : inputs) {
        net.set_input_data(kv.first, kv.second);
    }
    auto outs = net.execute();
    host_map r;
    for (const auto& id : out_ids) {
        auto m = outs.at(id).get_memory();
        r[id] = m->get_layout().data_type == data_types::f16 ? read_f16_as_f32(m) : read_f32(m);
    }
    return r;
}

void expect_outputs_match(const host_map& got, const host_map& ref, const std::string& tag) {
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
        std::cout << "[CMP " << tag << "] " << kv.first << " max_diff=" << max_diff << " mismatches=" << bad << "/" << g.size() << std::endl;
        EXPECT_EQ(bad, 0u) << tag << ": output " << kv.first << " does not match the reference (max_diff=" << max_diff << ")";
    }
}

void expect_outputs_equal(const host_map& a, const host_map& b, const std::string& tag) {
    for (const auto& kv : a) {
        const auto& o = b.at(kv.first);
        ASSERT_EQ(kv.second.size(), o.size());
        size_t bad = 0;
        for (size_t i = 0; i < o.size(); i++) {
            if (kv.second[i] != o[i]) {
                bad++;
            }
        }
        std::cout << "[EQ " << tag << "] " << kv.first << " differing=" << bad << "/" << o.size() << std::endl;
        EXPECT_EQ(bad, 0u) << tag << ": output " << kv.first << " differs between runs";
    }
}

// The consistency chain. `expect_sum` is the expected (post-fix) value of every link.
void check_chain(network& net, const primitive_id& w_id, const primitive_id& dep_id, bool expect_sum, const fused_primitive_desc& desc) {
    auto& eng = get_test_engine();
    auto& w = net.get_program()->get_node(w_id);
    const bool frozen = frozen_sum(w);
    const bool live = fusing::get_add_fusing_type(w, desc) == add_fusing_type::sum;
    const bool reused = fusing::get_reused_eltwmem_idx(w) != -1;
    const bool same = same_buffer(eng, net.get_primitive(w_id)->output_memory_ptr(), net.get_primitive(dep_id)->output_memory_ptr());
    std::cout << "[CHAIN] frozen_sum=" << frozen << " live_sum=" << live << " reused=" << reused << " same_buffer=" << same
              << " (expected " << expect_sum << ")" << std::endl;
    EXPECT_EQ(frozen, live) << "frozen/live disagree";
    EXPECT_EQ(live, reused) << "live/reused_idx disagree";
    EXPECT_EQ(reused, same) << "reused_idx/shared buffer disagree";
    EXPECT_EQ(frozen, expect_sum) << "frozen sum differs from expectation";
    EXPECT_EQ(live, expect_sum);
    if (!expect_sum) {
        EXPECT_EQ(fusing::get_reused_eltwmem_idx(w), -1);
        EXPECT_TRUE(fusing::get_add_fusing_type(w, desc) == add_fusing_type::binary_per_tensor);
    }
}

enum class dep_kind { input, view_of_input, strided_slice_view_of_input, mutable_data_dep, read_value_dep, internal, output };

// W = FC(in1) (oneDNN), S = sum(W, DEP), Z = FC(S) (ocl, keeps W at one user), out_Z = reorder(Z, f32).
struct f6_graph {
    topology t;
    ov::intel_gpu::ImplForcingMap forced;
    std::vector<std::string> outs;
    std::vector<std::string> custom_outputs;
    std::string dep_id;
};

f6_graph build_f6(engine& eng, dep_kind kind, memory::ptr md_mem) {
    f6_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(data("wW", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    switch (kind) {
    case dep_kind::input:
        t.add(input_layout("in2", layout{kShape, data_types::f16, format::bfyx}));
        g.dep_id = "in2";
        break;
    case dep_kind::view_of_input:
        t.add(input_layout("in2", layout{ov::PartialShape{1, 32, 16}, data_types::f16, format::bfyx}));
        t.add(reshape("Rs", input_info("in2"), false, std::vector<int64_t>{32, 16}, kShape));
        g.dep_id = "Rs";
        break;
    case dep_kind::strided_slice_view_of_input:
        t.add(input_layout("in2", layout{kShape, data_types::f16, format::bfyx}));
        t.add(strided_slice("Ss", input_info("in2"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
        g.dep_id = "Ss";
        break;
    case dep_kind::mutable_data_dep:
        t.add(mutable_data("m", md_mem));
        g.dep_id = "m";
        break;
    case dep_kind::read_value_dep:
        t.add(read_value{"rv", {input_info("in1")}, "v0", {layout{kShape, data_types::f16, format::bfyx}}});
        g.dep_id = "rv";
        break;
    case dep_kind::internal:
    case dep_kind::output:
        t.add(data("wD", make_weights(eng, 16, 16, 1)));
        t.add(fully_connected("D", input_info("in1"), "wD"));
        g.forced.emplace("D", impl_desc(impl_types::ocl));
        g.dep_id = "D";
        break;
    }
    t.add(eltwise("S", {input_info("W"), input_info(g.dep_id)}, eltwise_mode::sum));
    t.add(data("wZ", make_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("S"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    g.outs.push_back("out_Z");
    if (kind == dep_kind::output) {
        g.outs.push_back("D");
        g.custom_outputs = {"D", "out_Z"};
    }
    return g;
}

void run_f6(dep_kind kind, bool expect_sum) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN sum needs supports_immad";
    }
    const bool has_in2 = kind == dep_kind::input || kind == dep_kind::view_of_input || kind == dep_kind::strided_slice_view_of_input;
    const ov::PartialShape in2_shape = kind == dep_kind::view_of_input ? ov::PartialShape{1, 32, 16} : kShape;

    auto md_mem = make_f16(engine, kShape, 3);
    auto md_ref = make_f16(engine, kShape, 3);
    const auto md_snapshot = kind == dep_kind::mutable_data_dep ? read_f16_as_f32(md_mem) : std::vector<float>{};

    auto g = build_f6(engine, kind, md_mem);

    ExecutionConfig config = opt_config(engine, g.forced, g.custom_outputs);
    ASSERT_EQ(config.get_queue_type(), QueueTypes::in_order);
    auto prog = program::build_program(engine, g.t, config);
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << flags;

    // ---- graph-shape preconditions (stop and report if any fails; do not retune the topology)
    auto& W = prog->get_node("W");
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    const auto* desc = fused_sum(W);
    ASSERT_NE(desc, nullptr) << "W has no fused eltwise sum\n" << flags;
    ASSERT_EQ(W.get_dependency(desc->outer_dep_start_idx).id(), g.dep_id) << flags;
    auto& DEP = prog->get_node(g.dep_id);
    if (kind == dep_kind::view_of_input || kind == dep_kind::strided_slice_view_of_input) {
        ASSERT_TRUE(DEP.can_be_optimized() || DEP.is_runtime_skippable()) << flags;
    }
    if (kind == dep_kind::output) {
        ASSERT_TRUE(DEP.is_output()) << flags;
    }
    if (kind == dep_kind::read_value_dep) {
        std::cout << "[INFO] rv can_be_optimized=" << DEP.can_be_optimized() << std::endl;
    }
    if (kind != dep_kind::read_value_dep) {
        ASSERT_EQ(DEP.get_users().size(), 1u) << flags;
    }
    // vacuity guard: the pre-fix predicate accepts this sum
    ASSERT_TRUE(fusing::get_add_fusing_type(W, *desc, fusing::add_fusing_query::guard_superset) == add_fusing_type::sum)
        << "pre-fix predicate (guard_superset) does not say sum: the test would pass vacuously\n" << flags;

    // The reference graph is the same topology built without optimizations.
    auto gref = build_f6(engine, kind, md_ref);

    // Chain is compared BEFORE the first set_input_data (set_data swaps the input's memory afterwards).
    check_chain(net, "W", g.dep_id, expect_sum, *desc);

    std::map<std::string, memory::ptr> inA{{"in1", make_f16(engine, kShape, 1)}};
    std::map<std::string, memory::ptr> inB{{"in1", inA.at("in1")}};
    std::vector<float> snapA, snapB;
    if (has_in2) {
        inA.emplace("in2", make_f16(engine, in2_shape, 2));
        inB.emplace("in2", make_f16(engine, in2_shape, 4));
        snapA = read_f16_as_f32(inA.at("in2"));
        snapB = read_f16_as_f32(inB.at("in2"));
    }

    if (kind == dep_kind::read_value_dep) {
        layout vl{kShape, data_types::f16, format::bfyx};
        auto context = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
        // The variable buffer is allocated by the context's own engine; a oneDNN binary post-op reads it, so that
        // engine needs a oneDNN handle too (the test engine has one, a context created here does not).
        context->get_engine().create_onednn_engine(config);
        auto variable = std::make_shared<ov::intel_gpu::VariableState>(ov::intel_gpu::VariableStateInfo{"v0", vl}, context, net.get_shape_predictor());
        net.set_variable("v0", variable);
    }

    auto res1 = run_net(net, inA, g.outs);
    auto res2 = run_net(net, inB, g.outs);

    if (kind == dep_kind::read_value_dep) {
        // No assign and identical inputs: every run must start from the same variable state. A sum that wrote into the
        // variable buffer makes run 2 read what run 1 produced. If rv is not optimised the sum is rejected
        // conservatively and only the structural assertions above are meaningful.
        expect_outputs_equal(res1, res2, "rv run1 vs run2");
    } else {
        network net_ref(engine, gref.t, ref_config(engine, gref.custom_outputs));
        auto ref1 = run_net(net_ref, inA, gref.outs);
        auto ref2 = run_net(net_ref, inB, gref.outs);
        expect_outputs_match(res1, ref1, "run1");
        expect_outputs_match(res2, ref2, "run2");
    }

    // user data must be untouched
    if (has_in2) {
        EXPECT_EQ(read_f16_as_f32(inA.at("in2")), snapA) << "user input A was modified";
        EXPECT_EQ(read_f16_as_f32(inB.at("in2")), snapB) << "user input B was modified";
    }
    if (kind == dep_kind::mutable_data_dep) {
        EXPECT_EQ(read_f16_as_f32(md_mem), md_snapshot) << "user-attached mutable_data was modified";
    }
}

// ---------------------------------------------------------------- predicate-level graphs (no execution)

enum class misc_variant { md_output, md_control, strided_slice_view, output_view_member };

struct misc_graph {
    topology t;
    ov::intel_gpu::ImplForcingMap forced;
    std::vector<std::string> custom_outputs;
};

// D = FC(in1) (ocl); W = FC(...) (oneDNN); S = sum(W, D) is fused into W.
misc_graph build_misc(engine& eng, misc_variant v) {
    misc_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(data("wD", make_weights(eng, 16, 16, 1)));
    t.add(fully_connected("D", input_info("in1"), "wD"));
    std::string w_in = "in1";
    switch (v) {
    case misc_variant::md_output:
    case misc_variant::md_control:
        break;
    case misc_variant::strided_slice_view:
        t.add(strided_slice("V", input_info("D"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
        w_in = "V";
        break;
    case misc_variant::output_view_member:
        t.add(activation("X", input_info("D"), activation_func::relu));
        w_in = "X";
        break;
    }
    t.add(data("wW", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info(w_in), "wW"));
    t.add(eltwise("S", {input_info("W"), input_info("D")}, eltwise_mode::sum));
    g.forced.emplace("D", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    // Z keeps W at a single user and stops the output reorder from fusing into W.
    auto add_consumer = [&]() {
        t.add(data("wZ", make_weights(eng, 16, 16, 2)));
        t.add(fully_connected("Z", input_info("S"), "wZ"));
        t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
        g.forced.emplace("Z", impl_desc(impl_types::ocl));
    };
    if (v == misc_variant::md_output) {
        t.add(mutable_data("md", {input_info("S")}, make_f16(eng, kShape, 3)));
        g.custom_outputs = {"md"};
    } else if (v == misc_variant::output_view_member) {
        t.add(reshape("outV", input_info("D"), false, std::vector<int64_t>{1, 32, 16}, ov::PartialShape{1, 32, 16}));
        add_consumer();
        g.custom_outputs = {"outV", "out_Z"};
    } else {
        add_consumer();
    }
    return g;
}

// Builds the program and returns W (the sum writer) with its fused sum; asserts the vacuity guard.
void get_misc_writer(engine& eng, misc_variant v, program::ptr& prog, program_node*& w, const fused_primitive_desc*& desc) {
    auto g = build_misc(eng, v);
    prog = program::build_program(eng, g.t, opt_config(eng, g.forced, g.custom_outputs));
    const std::string flags = dump_flags(*prog);
    std::cout << flags;
    w = &prog->get_node("W");
    ASSERT_EQ(w->get_preferred_impl_type(), impl_types::onednn) << flags;
    desc = fused_sum(*w);
    ASSERT_NE(desc, nullptr) << "W has no fused eltwise sum\n" << flags;
    ASSERT_EQ(w->get_dependency(desc->outer_dep_start_idx).id(), "D") << flags;
    // For md_output W is already an output after build_program (basic_memory_dependencies ran); the test resets the flag
    // to the freeze-time state and checks the vacuity guard itself.
    if (v != misc_variant::md_output) {
        ASSERT_TRUE(fusing::get_add_fusing_type(*w, *desc, fusing::add_fusing_query::guard_superset) == add_fusing_type::sum)
            << "pre-fix predicate (guard_superset) does not say sum: the test would pass vacuously\n" << flags;
    }
}

// ---------------------------------------------------------------- follow-up: cascaded concat slot

memory::ptr make_conv_weights(engine& eng, size_t O, size_t I, size_t K) {
    layout l{ov::PartialShape{static_cast<int64_t>(O), static_cast<int64_t>(I), static_cast<int64_t>(K), static_cast<int64_t>(K)},
             data_types::f16, format::bfyx};
    auto mem = eng.allocate_memory(l);
    std::vector<ov::float16> w(O * I * K * K);
    for (size_t o = 0; o < O; o++) {
        for (size_t i = 0; i < I; i++) {
            for (size_t ky = 0; ky < K; ky++) {
                for (size_t kx = 0; kx < K; kx++) {
                    const bool centre = (ky == K / 2) && (kx == K / 2);
                    const float v = (o == i && centre) ? 0.5f : (((o + 2 * i + ky + kx) % 7 == 0) ? 0.125f : 0.f);
                    w[((o * I + i) * K + ky) * K + kx] = ov::float16(v);
                }
            }
        }
    }
    set_values(mem, w);
    return mem;
}

memory::ptr make_img(engine& eng, size_t C, int variant) {
    layout l{ov::PartialShape{1, static_cast<int64_t>(C), 32, 32}, data_types::f16, format::bfyx};
    auto mem = eng.allocate_memory(l);
    std::vector<ov::float16> v(l.count());
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = ov::float16(variant == 1 ? static_cast<float>(i % 17) * 0.125f - 1.0f
                                        : static_cast<float>((i * 3) % 13) * 0.0625f - 0.5f);
    }
    set_values(mem, v);
    return mem;
}

void add_conv(topology& t, const primitive_id& id, const primitive_id& in, const primitive_id& w, int pad) {
    t.add(convolution(id, input_info(in), w, "", 1, {1, 1}, {1, 1}, {pad, pad}, {pad, pad}, false));
}

ov::intel_gpu::ImplementationDesc conv_desc(impl_types t) {
    return ov::intel_gpu::ImplementationDesc{format::b_fs_yx_fsv16, "", t};
}

// P, Q -> C1 = concat(P, Q); Q3; C2 = concat(C1, Q3); W = conv(in2) with S = sum(W, C2); R = prod(P, Z) reads P late.
struct cascade_graph {
    topology t;
    ov::intel_gpu::ImplForcingMap forced;
    std::vector<std::string> outs;
};

cascade_graph build_cascade(engine& eng) {
    cascade_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{ov::PartialShape{1, 16, 32, 32}, data_types::f16, format::bfyx}));
    t.add(data("w_pq", make_conv_weights(eng, 16, 16, 1)));
    add_conv(t, "P", "in1", "w_pq", 0);
    add_conv(t, "Q", "in1", "w_pq", 0);
    add_conv(t, "Q3", "in1", "w_pq", 0);
    t.add(concatenation("C1", {input_info("P"), input_info("Q")}, 1));
    t.add(concatenation("C2", {input_info("C1"), input_info("Q3")}, 1));
    t.add(input_layout("in2", layout{ov::PartialShape{1, 48, 32, 32}, data_types::f16, format::bfyx}));
    t.add(data("w_w", make_conv_weights(eng, 48, 48, 3)));
    add_conv(t, "W", "in2", "w_w", 1);
    t.add(eltwise("S", {input_info("W"), input_info("C2")}, eltwise_mode::sum));
    t.add(reorder("out_S", input_info("S"), format::bfyx, data_types::f32));
    t.add(data("w_z", make_conv_weights(eng, 16, 48, 1)));
    add_conv(t, "Z", "S", "w_z", 0);
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    t.add(eltwise("R", {input_info("P"), input_info("Z")}, eltwise_mode::prod));
    t.add(reorder("out_R", input_info("R"), format::bfyx, data_types::f32));
    g.outs = {"out_S", "out_Z", "out_R"};
    g.forced = {{"P", conv_desc(impl_types::ocl)}, {"Q", conv_desc(impl_types::ocl)}, {"Q3", conv_desc(impl_types::ocl)},
                {"C1", conv_desc(impl_types::ocl)}, {"C2", conv_desc(impl_types::ocl)}, {"Z", conv_desc(impl_types::ocl)},
                {"R", conv_desc(impl_types::ocl)}, {"W", conv_desc(impl_types::onednn)}};
    return g;
}

}  // namespace

// ---------------------------------------------------------------- F13

// W sums into a single-user input_layout. This is the ARBITER of the F13 analysis (it reverses upstream ceedce643f).
TEST(inplace_alias_conflict_f6_f13, F13_sum_into_single_user_input) {
    run_f6(dep_kind::input, /*expect_sum=*/false);
}

// ---------------------------------------------------------------- F6

TEST(inplace_alias_conflict_f6_f13, F6a_sum_dep_is_network_output) {
    run_f6(dep_kind::output, /*expect_sum=*/false);
}

// Sum dep is a reshape view of a user input: the sum must be rejected (the view's owner is an external input_layout).
TEST(inplace_alias_conflict_f6_f13, F6b_sum_dep_view_of_input) {
    run_f6(dep_kind::view_of_input, /*expect_sum=*/false);
}

TEST(inplace_alias_conflict_f6_f13, F6e_sum_dep_static_strided_slice_view_of_input) {
    run_f6(dep_kind::strided_slice_view_of_input, /*expect_sum=*/false);
}

TEST(inplace_alias_conflict_f6_f13, F6c_sum_dep_mutable_data) {
    run_f6(dep_kind::mutable_data_dep, /*expect_sum=*/false);
}

TEST(inplace_alias_conflict_f6_f13, F6d_sum_dep_read_value) {
    run_f6(dep_kind::read_value_dep, /*expect_sum=*/false);
}

// Control: an internal single-user dep that is not an output keeps sum (the fix must not over-reject).
TEST(inplace_alias_conflict_f6_f13, F6_control_internal_dep) {
    run_f6(dep_kind::internal, /*expect_sum=*/true);
}

// ---------------------------------------------------------------- follow-up: cascaded concat slot

// F10 through nested in-place concats: the sum dependency C2 holds C1, whose slot P has a late reader R.
TEST(inplace_alias_conflict_f6_f13, F10_cascaded_concat_slot_late_reader) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN sum needs supports_immad";
    }
    auto g = build_cascade(engine);
    std::map<std::string, memory::ptr> inputs{{"in1", make_img(engine, 16, 1)}, {"in2", make_img(engine, 48, 2)}};

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(g.forced));
    ASSERT_EQ(config.get_queue_type(), QueueTypes::in_order);
    auto prog = program::build_program(engine, g.t, config);
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << flags;

    auto& P = prog->get_node("P");
    auto& C1 = prog->get_node("C1");
    auto& C2 = prog->get_node("C2");
    auto& W = prog->get_node("W");
    ASSERT_TRUE(C1.can_be_optimized()) << flags;
    ASSERT_TRUE(C2.can_be_optimized()) << flags;
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    ASSERT_EQ(P.get_users().size(), 2u) << flags;
    const auto* desc = fused_sum(W);
    ASSERT_NE(desc, nullptr) << flags;
    ASSERT_EQ(W.get_dependency(desc->outer_dep_start_idx).id(), "C2") << flags;
    ASSERT_TRUE(fusing::get_add_fusing_type(W, *desc, fusing::add_fusing_query::guard_superset) == add_fusing_type::sum) << flags;

    auto res = run_net(net, inputs, g.outs);
    auto gref = build_cascade(engine);
    ExecutionConfig rc = get_test_default_config(engine);
    rc.set_property(ov::intel_gpu::optimize_data(false));
    network net_ref(engine, gref.t, rc);
    auto ref = run_net(net_ref, inputs, gref.outs);

    check_chain(net, "W", "C2", /*expect_sum=*/false, *desc);
    expect_outputs_match(res, ref, "cascade");
}

// ---------------------------------------------------------------- A3 / follow-ups (predicate level)

// A3: W's only user is an output mutable_data, so W becomes an output in basic_memory_dependencies, AFTER the freeze. The
// exact predicate must already reject the sum at freeze time (flag still unset) and keep rejecting afterwards.
TEST(inplace_alias_conflict_f6_f13, F6_A3_writer_becomes_output_exact_rejects_before_and_after_bmd) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "a sum with an outer dependency is only fused with oneDNN impls enabled";
    }
    program::ptr prog;
    program_node* w = nullptr;
    const fused_primitive_desc* desc = nullptr;
    ASSERT_NO_FATAL_FAILURE(get_misc_writer(engine, misc_variant::md_output, prog, w, desc));
    ASSERT_TRUE(prog->get_node("md").is_output());
    ASSERT_EQ(w->get_users().size(), 1u);
    ASSERT_EQ(w->get_users().front()->id(), "md");
    // build_program already ran basic_memory_dependencies: emulate the freeze-time state (flag not yet propagated).
    ASSERT_TRUE(w->is_output()) << "basic_memory_dependencies should have propagated the output flag";
    w->set_output(false);
    ASSERT_TRUE(fusing::get_add_fusing_type(*w, *desc, fusing::add_fusing_query::guard_superset) == add_fusing_type::sum)
        << "pre-fix predicate (guard_superset) does not say sum at freeze time: the test would pass vacuously";
    EXPECT_TRUE(fusing::get_add_fusing_type(*w, *desc) != add_fusing_type::sum)
        << "freeze-time exact predicate accepts a sum although the writer becomes a network output";
    program_wrapper::apply_opt_pass<basic_memory_dependencies>(*prog);
    EXPECT_TRUE(w->is_output());
    EXPECT_TRUE(fusing::get_add_fusing_type(*w, *desc) != add_fusing_type::sum)
        << "frozen/live divergence: exact predicate changed after basic_memory_dependencies";
}

// Control: the same graph without the output mutable_data keeps sum (no over-rejection).
TEST(inplace_alias_conflict_f6_f13, F6_A3_control_writer_not_output_keeps_sum) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "a sum with an outer dependency is only fused with oneDNN impls enabled";
    }
    program::ptr prog;
    program_node* w = nullptr;
    const fused_primitive_desc* desc = nullptr;
    ASSERT_NO_FATAL_FAILURE(get_misc_writer(engine, misc_variant::md_control, prog, w, desc));
    EXPECT_TRUE(fusing::get_add_fusing_type(*w, *desc) == add_fusing_type::sum);
}

// Follow-up of F10/F11: the writer reads the sum dependency through a static no-op strided_slice (a post-freeze view).
// (A view through a plain reorder cannot be built at predicate level: a surviving reorder changes dtype/format, which
// already forbids the sum, and an identical one is removed.)
TEST(inplace_alias_conflict_f6_f13, F11_self_read_through_static_strided_slice_view) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "a sum with an outer dependency is only fused with oneDNN impls enabled";
    }
    program::ptr prog;
    program_node* w = nullptr;
    const fused_primitive_desc* desc = nullptr;
    ASSERT_NO_FATAL_FAILURE(get_misc_writer(engine, misc_variant::strided_slice_view, prog, w, desc));
    auto& V = prog->get_node("V");
    ASSERT_TRUE(V.can_be_optimized() || V.is_runtime_skippable()) << dump_flags(*prog);
    EXPECT_TRUE(fusing::get_add_fusing_type(*w, *desc) != add_fusing_type::sum)
        << "writer reads the sum dependency through a view (strided_slice) but sum was accepted";
}

// ---------------------------------------------------------------- Step 7 follow-ups: path nodes, md cascade

namespace {

enum class view_owner_variant { owner_is_output, control_internal };

// O = FC(in3) (ocl, internal producer); V = no-op strided_slice(O) is a view of O; W = FC(in1) (oneDNN); S = sum(W, V) fused into W.
struct view_owner_graph {
    topology t;
    ov::intel_gpu::ImplForcingMap forced;
    std::vector<std::string> outs;
    std::vector<std::string> custom_outputs;
};

view_owner_graph build_view_owner(engine& eng, view_owner_variant v) {
    view_owner_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
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
    if (v == view_owner_variant::owner_is_output) {
        g.outs.push_back("O");
        g.custom_outputs = {"O", "out_Z"};
    }
    return g;
}

void run_view_owner(view_owner_variant v, bool expect_sum, bool check_runtime = true) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN sum needs supports_immad";
    }
    auto g = build_view_owner(engine, v);
    auto prog = program::build_program(engine, g.t, opt_config(engine, g.forced, g.custom_outputs));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << flags;
    auto& W = prog->get_node("W");
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    const auto* desc = fused_sum(W);
    ASSERT_NE(desc, nullptr) << "W has no fused eltwise sum\n" << flags;
    ASSERT_EQ(W.get_dependency(desc->outer_dep_start_idx).id(), "V") << flags;
    auto& V = prog->get_node("V");
    ASSERT_TRUE(V.can_be_optimized() || V.is_runtime_skippable()) << flags;
    if (v == view_owner_variant::owner_is_output) {
        ASSERT_TRUE(prog->get_node("O").is_output()) << flags;
        ASSERT_FALSE(V.is_output()) << flags;
    }
    // vacuity guard: the pre-fix predicate accepts this sum
    ASSERT_TRUE(fusing::get_add_fusing_type(W, *desc, fusing::add_fusing_query::guard_superset) == add_fusing_type::sum)
        << "pre-fix predicate (guard_superset) does not say sum: the test would pass vacuously\n" << flags;
    if (!check_runtime) {
        // predicate level only: frozen, live and reuse index agree (see the F6h test for the runtime side)
        EXPECT_EQ(frozen_sum(W), expect_sum);
        EXPECT_EQ(fusing::get_add_fusing_type(W, *desc) == add_fusing_type::sum, expect_sum);
        EXPECT_EQ(fusing::get_reused_eltwmem_idx(W) != -1, expect_sum);
        return;
    }
    check_chain(net, "W", "V", expect_sum, *desc);

    std::map<std::string, memory::ptr> in{{"in1", make_f16(engine, kShape, 1)}, {"in3", make_f16(engine, kShape, 2)}};
    auto res = run_net(net, in, g.outs);
    auto gref = build_view_owner(engine, v);
    network net_ref(engine, gref.t, ref_config(engine, gref.custom_outputs));
    auto ref = run_net(net_ref, in, gref.outs);
    expect_outputs_match(res, ref, "view_owner");
}

}  // namespace

// F6f: the producer O (path node after the queried dep, not the dep itself) is a network output; dep V is a (no-op strided_slice) view of O.
TEST(inplace_alias_conflict_f6_f13, F6f_path_non_first_node_is_output) {
    run_view_owner(view_owner_variant::owner_is_output, /*expect_sum=*/false);
}

// Control: the same view of an internal, non-output producer keeps the sum (over-rejection guard). Pre-fix the sum is
// fused here too (guard_superset says sum), so the control is not vacuous.
TEST(inplace_alias_conflict_f6_f13, F6_control_internal_view_keeps_sum) {
    run_view_owner(view_owner_variant::control_internal, /*expect_sum=*/true, /*check_runtime=*/false);
}

// F19 (test name keeps the historical F6h): when the sum dependency is a view (no-op strided_slice) of an internal producer,
// W's dst was bound at construction to the view's private buffer and never re-pointed. Fixed by
// primitive_inst::rebind_reused_eltw_output() (called from network::allocate_primitives). See the F19 design doc.
TEST(inplace_alias_conflict_f6_f13, F6h_sum_into_view_of_internal_producer_runtime) {
    run_view_owner(view_owner_variant::control_internal, /*expect_sum=*/true, /*check_runtime=*/true);
}

// F6g: mutable_data cascade S -> md1 -> md2 (output) hanging off the writer. basic_memory_dependencies propagates only one
// level, so at freeze time neither W nor md1 is flagged; is_or_becomes_output must recurse through md1.
TEST(inplace_alias_conflict_f6_f13, F6g_md_cascade_writer_becomes_output) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "a sum with an outer dependency is only fused with oneDNN impls enabled";
    }
    auto g = build_misc(engine, misc_variant::md_control);
    auto& t = g.t;
    t.add(mutable_data("md1", {input_info("S")}, make_f16(engine, kShape, 3)));
    t.add(mutable_data("md2", {input_info("md1")}, make_f16(engine, kShape, 4)));
    g.custom_outputs = {"md2"};
    auto prog = program::build_program(engine, g.t, opt_config(engine, g.forced, g.custom_outputs));
    const std::string flags = dump_flags(*prog);
    std::cout << flags;
    auto* w = &prog->get_node("W");
    ASSERT_EQ(w->get_preferred_impl_type(), impl_types::onednn) << flags;
    const auto* desc = fused_sum(*w);
    ASSERT_NE(desc, nullptr) << flags;
    ASSERT_EQ(w->get_dependency(desc->outer_dep_start_idx).id(), "D") << flags;
    auto& md1 = prog->get_node("md1");
    auto& md2 = prog->get_node("md2");
    ASSERT_TRUE(md2.is_output()) << flags;
    ASSERT_EQ(w->get_users().size(), 1u) << flags;
    ASSERT_EQ(w->get_users().front()->id(), "md1") << flags;
    ASSERT_EQ(md1.get_users().front()->id(), "md2") << flags;
    // emulate the freeze-time state (the flags are propagated by the already executed basic_memory_dependencies)
    w->set_output(false);
    md1.set_output(false);
    EXPECT_TRUE(program_helpers::is_or_becomes_output(*w));
    ASSERT_TRUE(fusing::get_add_fusing_type(*w, *desc, fusing::add_fusing_query::guard_superset) == add_fusing_type::sum)
        << "pre-fix predicate (guard_superset) does not say sum at freeze time: the test would pass vacuously";
    EXPECT_TRUE(fusing::get_add_fusing_type(*w, *desc) != add_fusing_type::sum)
        << "exact predicate accepts a sum although the writer becomes an output through a mutable_data cascade";
}

// F6i: the sum dependency is a condition output. condition re-points its output buffer on every execution, so a dst
// bound at construction would go stale; exact must reject (is_external_buffer_root covers condition).
TEST(inplace_alias_conflict_f6_f13, F6i_sum_dep_is_condition_output) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN sum needs supports_immad";
    }
    auto build = [&](bool for_ref) {
        ExecutionConfig bcfg = get_test_default_config(engine);
        auto mk_branch = [&](const std::string& out_id, activation_func f) {
            topology bt;
            bt.add(input_layout("bin", layout{kShape, data_types::f16, format::bfyx}));
            bt.add(activation(out_id, input_info("bin"), f));
            condition::branch b;
            b.inner_program = program::build_program(engine, bt, bcfg, false, false, true);
            b.input_map.insert({"in2", "bin"});
            b.output_map.insert({0, out_id});
            return b;
        };
        topology t;
        t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
        t.add(input_layout("in2", layout{kShape, data_types::f16, format::bfyx}));
        t.add(input_layout("pred", layout{ov::PartialShape{1}, data_types::u8, format::bfyx}));
        t.add(condition("C", {input_info("pred"), input_info("in2")}, mk_branch("bt", activation_func::relu), mk_branch("bf", activation_func::abs)));
        t.add(data("wW", make_weights(engine, 16, 16, 0)));
        t.add(fully_connected("W", input_info("in1"), "wW"));
        t.add(eltwise("S", {input_info("W"), input_info("C")}, eltwise_mode::sum));
        t.add(data("wZ", make_weights(engine, 16, 16, 2)));
        t.add(fully_connected("Z", input_info("S"), "wZ"));
        t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
        return t;
    };
    ov::intel_gpu::ImplForcingMap forced;
    forced.emplace("W", impl_desc(impl_types::onednn));
    forced.emplace("Z", impl_desc(impl_types::ocl));
    auto topo = build(false);
    auto prog = program::build_program(engine, topo, opt_config(engine, forced));
    const std::string flags = dump_flags(*prog);
    std::cout << flags;
    auto& W = prog->get_node("W");
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    const auto* desc = fused_sum(W);
    ASSERT_NE(desc, nullptr) << "W has no fused eltwise sum\n" << flags;
    ASSERT_EQ(W.get_dependency(desc->outer_dep_start_idx).id(), "C") << flags;
    ASSERT_TRUE(fusing::get_add_fusing_type(W, *desc, fusing::add_fusing_query::guard_superset) == add_fusing_type::sum)
        << "pre-fix predicate (guard_superset) does not say sum: the test would pass vacuously\n" << flags;
    EXPECT_TRUE(fusing::get_add_fusing_type(W, *desc) != add_fusing_type::sum) << "sum accepted into a condition output";
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W), -1);
    EXPECT_FALSE(frozen_sum(W));

    network net(prog, 0);
    // A fresh reference network per iteration: the reference is only trusted for one execution
    for (int pv : {0, 1}) {
        auto topo_ref = build(true);
        network net_ref(engine, topo_ref, ref_config(engine));
        auto pred = engine.allocate_memory(layout{ov::PartialShape{1}, data_types::u8, format::bfyx});
        set_values<uint8_t>(pred, {static_cast<uint8_t>(pv)});
        std::map<std::string, memory::ptr> in{{"in1", make_f16(engine, kShape, 1)}, {"in2", make_f16(engine, kShape, 2 + pv)}, {"pred", pred}};
        auto res = run_net(net, in, {"out_Z"});
        auto ref = run_net(net_ref, in, {"out_Z"});
        expect_outputs_match(res, ref, "condition pred=" + std::to_string(pv));
    }
}

// A4: DISABLED. Downstream output view members of the owner's class are not checked in Pre-0: D has an output view
// (reshape outV) that reads D's buffer after W has written the sum into it.
// Deferred to Phase 1a war_safe (virtual reader for output members).
// NOTE: as written the topology does not reach a fused sum (the vacuity guard in get_misc_writer fails because D has three
// users: outV, X and W), so it needs a redesign together with the Phase 1a war_safe work before it can be enabled.
TEST(inplace_alias_conflict_f6_f13, DISABLED_A4_downstream_output_view_member_of_owner_class) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "a sum with an outer dependency is only fused with oneDNN impls enabled";
    }
    program::ptr prog;
    program_node* w = nullptr;
    const fused_primitive_desc* desc = nullptr;
    ASSERT_NO_FATAL_FAILURE(get_misc_writer(engine, misc_variant::output_view_member, prog, w, desc));
    EXPECT_TRUE(fusing::get_add_fusing_type(*w, *desc) != add_fusing_type::sum);
}
