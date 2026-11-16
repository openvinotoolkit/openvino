// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression tests for
//   F8: a static no-op resample must not be marked skippable when it is a network output (F8a) or an in-place concat
//       predecessor (F8b). Every test asserts the CORRECT behavior. ocl only, no oneDNN needed.

#include "test_utils.h"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "openvino/op/interpolate.hpp"

#include "program_node.h"
#include "program_helpers.h"
#include "primitive_inst.h"

#include <iostream>
#include <map>
#include <sstream>

using namespace cldnn;
using namespace ::tests;

namespace {

std::vector<float> read_f32(memory::ptr mem) {
    mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
    return std::vector<float>(lock.begin(), lock.end());
}

std::vector<float> gen_values(size_t n, int variant) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = variant == 1 ? static_cast<float>(i % 17) * 0.125f - 1.0f : static_cast<float>((i * 3) % 13) * 0.0625f - 0.5f;
    }
    return v;
}

memory::ptr make_in(engine& eng, const layout& l, int variant) {
    auto mem = eng.allocate_memory(l);
    set_values(mem, gen_values(l.count(), variant));
    return mem;
}

ExecutionConfig opt_config(engine& eng, const ov::intel_gpu::ImplForcingMap& forced) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(forced));
    return config;
}

ExecutionConfig ref_config(engine& eng) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    return config;
}

ov::intel_gpu::ImplementationDesc bfyx_impl(impl_types t) {
    return ov::intel_gpu::ImplementationDesc{format::bfyx, "", t};
}

std::string dump_flags(const program& prog) {
    std::stringstream ss;
    for (const auto* n : prog.get_processing_order()) {
        ss << "  " << n->id() << " type=" << n->get_primitive()->type_string() << " opt=" << n->can_be_optimized()
           << " skippable=" << n->is_runtime_skippable() << " output=" << n->is_output() << " pad_up=";
        const auto& pad = n->get_output_layout(false).data_padding;
        for (size_t i = 0; i < 4; i++) {
            ss << pad._upper_size[i] << (i == 3 ? "" : ",");
        }
        ss << " deps=[";
        for (const auto& d : n->get_dependencies()) {
            ss << d.first->id() << " ";
        }
        ss << "] users=[";
        for (const auto* u : n->get_users()) {
            ss << u->id() << " ";
        }
        ss << "]\n";
    }
    return ss.str();
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
    ASSERT_EQ(got.size(), ref.size());
    for (const auto& kv : ref) {
        const auto& g = got.at(kv.first);
        ASSERT_EQ(g.size(), kv.second.size()) << kv.first;
        for (size_t i = 0; i < g.size(); i++) {
            ASSERT_NEAR(g[i], kv.second[i], 1e-2f) << kv.first << " @" << i;
        }
    }
}

// ----------------------------------------------------------------------------------------------- F8

topology build_f8a(bool compute_producer) {
    topology t;
    const layout l{ov::PartialShape{1, 16, 8, 8}, data_types::f32, format::bfyx};
    t.add(input_layout("in", l));
    std::string src = "in";
    if (compute_producer) {
        t.add(activation("P", input_info("in"), activation_func::abs));
        src = "P";
    }
    t.add(resample("R", input_info(src), tensor(1, 16, 8, 8), 1, ov::op::v4::Interpolate::InterpolateMode::NEAREST));
    return t;
}

void run_f8a(bool compute_producer) {
    auto& engine = get_test_engine();
    const layout l{ov::PartialShape{1, 16, 8, 8}, data_types::f32, format::bfyx};
    ov::intel_gpu::ImplForcingMap forced{{"R", bfyx_impl(impl_types::ocl)}};
    if (compute_producer) {
        forced.emplace("P", bfyx_impl(impl_types::ocl));
    }
    auto prog = program::build_program(engine, build_f8a(compute_producer), opt_config(engine, forced));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << flags;

    auto& R = prog->get_node("R");
    ASSERT_TRUE(R.is_output()) << flags;
    ASSERT_TRUE(R.is_type<resample>()) << flags;
    ASSERT_EQ(R.get_input_layout(0), R.get_output_layout(false)) << flags;
    EXPECT_FALSE(R.can_be_optimized()) << flags;
    EXPECT_FALSE(R.is_runtime_skippable()) << flags;

    auto a = make_in(engine, l, 1);
    const auto a_vals = read_f32(a);
    auto u = engine.allocate_memory(l);
    set_values(u, std::vector<float>(l.count(), -7.f));

    net.set_input_data("in", a);
    net.set_output_memory("R", u);
    auto outs = net.execute();
    net.get_stream().finish();  // the network runs on its own stream

    auto expected = a_vals;
    if (compute_producer) {
        for (auto& x : expected) {
            x = std::abs(x);
        }
    }
    EXPECT_EQ(read_f32(u), expected) << "user output memory was not written\n" << flags;
    EXPECT_EQ(read_f32(a), a_vals) << "graph input must stay unchanged\n" << flags;
}

topology build_f8b() {
    topology t;
    const layout l{ov::PartialShape{1, 16, 8, 8}, data_types::f32, format::bfyx};
    t.add(input_layout("in1", l));
    t.add(input_layout("in2", l));
    t.add(activation("P", input_info("in1"), activation_func::abs));
    t.add(resample("R", input_info("P"), tensor(1, 16, 8, 8), 1, ov::op::v4::Interpolate::InterpolateMode::NEAREST));
    t.add(activation("Q", input_info("in2"), activation_func::abs));
    t.add(concatenation("C", {input_info("R"), input_info("Q")}, 1));
    t.add(activation("A", input_info("C"), activation_func::abs));
    t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
    return t;
}

}  // namespace

// ---------------------------------------------------------------- F8 (ocl only)

// Pre-fix: a no-op output resample is skipped and resample_inst::update_output_memory re-aliases _outputs[0] to the
// input on every execution, discarding the user's set_output_memory buffer. Plugin host outputs are rescued by the
// copy-back in sync_infer_request; remote outputs and direct set_output_memory callers are not.
TEST(inplace_alias_conflict_f8, F8a_output_noop_resample_of_input) {
    run_f8a(false);
}

// Flag-level guard. With a compute producer, set_output_memory rebinds the output chain, which includes the producer,
// so the data was already correct before the fix; this test keeps resample consistent with the other skippable types
// (an output owns its buffer) at the cost of one copy. Real data loss happens only when the producer is a graph input
// or an external buffer: a lazily allocated input_layout is not in the output chain (network.cpp build_output_chain,
// outputs_allocated()), so resample_inst::update_output_memory re-aliases R to the input when the buffers differ, and
// the bound memory is dropped (see F8a_output_noop_resample_of_input).
TEST(inplace_alias_conflict_f8, F8a_output_noop_resample_of_compute_producer) {
    run_f8a(true);
}

// Defense-in-depth guard (not a reproduction of a known bug): a static no-op resample that feeds an in-place concat
// must stay a real kernel, so that its slot is written. Must hold before and after the fix.
TEST(inplace_alias_conflict_f8, F8b_noop_resample_into_inplace_concat_guard) {
    auto& engine = get_test_engine();
    const layout l{ov::PartialShape{1, 16, 8, 8}, data_types::f32, format::bfyx};
    ov::intel_gpu::ImplForcingMap forced;
    for (const char* id : {"P", "R", "Q", "C", "A"}) {
        forced.emplace(id, bfyx_impl(impl_types::ocl));
    }
    auto prog = program::build_program(engine, build_f8b(), opt_config(engine, forced));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << flags;

    auto& R = prog->get_node("R");
    auto& C = prog->get_node("C");
    ASSERT_TRUE(C.can_be_optimized()) << flags;
    EXPECT_FALSE(R.can_be_optimized()) << flags;
    EXPECT_FALSE(R.is_runtime_skippable()) << flags;

    std::map<std::string, memory::ptr> inputs{{"in1", make_in(engine, l, 1)}, {"in2", make_in(engine, l, 2)}};
    auto res = run_net(net, inputs, {"out"});

    network net_ref(engine, build_f8b(), ref_config(engine));
    auto ref = run_net(net_ref, inputs, {"out"});
    expect_outputs_match(res, ref);
}
