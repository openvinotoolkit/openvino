// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression tests for
//   F12: a network-output permute must not be optimized out together with its reorder producer;
//   F14: a network-output read_value must not alias the variable buffer.
// Every test asserts the CORRECT behavior.

#include "test_utils.h"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/softmax.hpp"

#include "program_node.h"
#include "primitive_inst.h"

#include <cmath>

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

namespace {

std::vector<float> arange(size_t n, float start) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = start + static_cast<float>(i);
    }
    return v;
}

std::vector<float> read_f32(memory::ptr mem) {
    mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
    return std::vector<float>(lock.begin(), lock.end());
}

ExecutionConfig f12_config(engine& eng) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    return config;
}

// input(byfx) -> reorder(bfyx) -> permute{0,2,1,3} [-> softmax users]
topology make_f12_topology(bool with_users) {
    topology t;
    t.add(input_layout("X", layout{ov::PartialShape{1, 2, 3, 5}, data_types::f32, format::byfx}));
    t.add(reorder("r", input_info("X"), format::bfyx, data_types::f32));
    t.add(permute("p", input_info("r"), {0, 2, 1, 3}));
    if (with_users) {
        t.add(softmax("s", input_info("p"), 3));
        t.add(softmax("s2", input_info("p"), 1));
    }
    return t;
}

// Physical byfx-order bytes of X read back as bfyx {1,3,2,5}.
const std::vector<float>& vx() {
    static const std::vector<float> v = arange(30, 1.f);
    return v;
}

memory::ptr make_x(engine& eng, const std::vector<float>& vals) {
    auto mx = eng.allocate_memory(layout{ov::PartialShape{1, 2, 3, 5}, data_types::f32, format::byfx});
    set_values(mx, vals);
    return mx;
}

// Variable topology
topology make_f14_topology(const layout& vl) {
    topology t;
    t.add(input_layout("in", vl));
    t.add(read_value{"rv", {input_info("in")}, "v0", {vl}});
    t.add(eltwise{"sum", {input_info("in"), input_info("rv")}, eltwise_mode::sum, {}, data_types::f32});
    t.add(assign{"assign", {input_info("sum")}, "v0", vl});
    return t;
}

}  // namespace

// ---------------------------------------------------------------- F12

TEST(inplace_alias_conflict_f12_f14, f12_reorder_permute_output_not_optimized) {
    auto& engine = get_test_engine();
    auto prog = program::build_program(engine, make_f12_topology(false), f12_config(engine), false, false);
    ASSERT_NE(prog, nullptr);

    auto& p = prog->get_node("p");
    auto& r = prog->get_node("r");
    ASSERT_TRUE(p.is_output());
    EXPECT_FALSE(p.can_be_optimized());
    EXPECT_FALSE(r.can_be_optimized());
}

TEST(inplace_alias_conflict_f12_f14, f12_reorder_permute_output_does_not_alias_input) {
    auto& engine = get_test_engine();
    auto prog = program::build_program(engine, make_f12_topology(false), f12_config(engine), false, false);
    ASSERT_NE(prog, nullptr);
    network net(prog, 0);

    auto mx = make_x(engine, vx());
    net.set_input_data("X", mx);
    auto outs = net.execute();
    auto out = outs.at("p").get_memory();

    EXPECT_FALSE(engine.is_the_same_buffer(*out, *mx));
    EXPECT_EQ(read_f32(out), vx());

    set_values(mx, arange(30, 1001.f));
    EXPECT_EQ(read_f32(out), vx());
}

TEST(inplace_alias_conflict_f12_f14, f12_reorder_permute_output_honors_set_output_memory) {
    auto& engine = get_test_engine();
    auto prog = program::build_program(engine, make_f12_topology(false), f12_config(engine), false, false);
    ASSERT_NE(prog, nullptr);
    network net(prog, 0);

    auto om = engine.allocate_memory(prog->get_node("p").get_output_layout());
    set_values(om, std::vector<float>(30, -7.f));

    auto mx = make_x(engine, vx());
    net.set_input_data("X", mx);
    net.set_output_memory("p", om);
    auto outs = net.execute();

    EXPECT_TRUE(engine.is_the_same_buffer(*outs.at("p").get_memory(), *om));
    EXPECT_EQ(read_f32(om), vx());
}

// Guard: permute with non-fusing users (not an output) is still optimized together with the reorder.
TEST(inplace_alias_conflict_f12_f14, f12_reorder_permute_non_output_still_optimized) {
    auto& engine = get_test_engine();
    auto prog = program::build_program(engine, make_f12_topology(true), f12_config(engine), false, false);
    ASSERT_NE(prog, nullptr);

    auto& p = prog->get_node("p");
    auto& r = prog->get_node("r");
    ASSERT_FALSE(p.is_output());
    ASSERT_TRUE(p.get_fused_primitives().empty());
    EXPECT_TRUE(r.can_be_optimized());
    EXPECT_TRUE(p.can_be_optimized());

    network net(prog, 0);
    net.set_input_data("X", make_x(engine, vx()));
    auto outs = net.execute();
    auto out = read_f32(outs.at("s").get_memory());
    ASSERT_EQ(out.size(), 30u);
    // softmax along x (5 elements) of each row of vx
    for (size_t row = 0; row < 6; row++) {
        float denom = 0.f;
        for (size_t i = 0; i < 5; i++) {
            denom += std::exp(vx()[row * 5 + i] - vx()[row * 5 + 4]);
        }
        for (size_t i = 0; i < 5; i++) {
            float expected = std::exp(vx()[row * 5 + i] - vx()[row * 5 + 4]) / denom;
            EXPECT_NEAR(out[row * 5 + i], expected, 1e-4f) << "row " << row << " i " << i;
        }
    }
}

// ---------------------------------------------------------------- F14

TEST(inplace_alias_conflict_f12_f14, f14_read_value_output_not_optimized_static) {
    auto& engine = get_test_engine();
    layout vl{ov::PartialShape{1, 1, 2, 2}, data_types::f32, format::bfyx};
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"rv", "assign"}));
    auto prog = program::build_program(engine, make_f14_topology(vl), config, false, false);
    ASSERT_NE(prog, nullptr);

    auto& rv = prog->get_node("rv");
    ASSERT_TRUE(rv.is_output());
    EXPECT_FALSE(rv.can_be_optimized());
}

TEST(inplace_alias_conflict_f12_f14, f14_read_value_output_returns_pre_assign_state) {
    auto& engine = get_test_engine();
    layout vl{ov::PartialShape{1, 1, 2, 2}, data_types::f32, format::bfyx};
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"rv", "assign"}));

    network net(engine, make_f14_topology(vl), config);
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto variable = std::make_shared<VariableState>(VariableStateInfo{"v0", vl}, context, net.get_shape_predictor());
    net.set_variable("v0", variable);

    const std::vector<float> vals = {1.f, 2.f, 3.f, 4.f};
    auto m = engine.allocate_memory(vl);
    set_values(m, vals);
    net.set_input_data("in", m);

    for (int k = 1; k <= 3; k++) {
        auto outs = net.execute();
        auto rv_out = read_f32(outs.at("rv").get_memory());
        auto as_out = read_f32(outs.at("assign").get_memory());
        ASSERT_EQ(rv_out.size(), vals.size());
        ASSERT_EQ(as_out.size(), vals.size());
        for (size_t i = 0; i < vals.size(); i++) {
            EXPECT_FLOAT_EQ(rv_out[i], k * vals[i]) << "run " << k << " rv[" << i << "]";
            EXPECT_FLOAT_EQ(as_out[i], (k + 1) * vals[i]) << "run " << k << " assign[" << i << "]";
        }
    }
}

// Guard: a non-output single-user read_value stays optimized.
TEST(inplace_alias_conflict_f12_f14, f14_read_value_non_output_single_user_still_optimized) {
    auto& engine = get_test_engine();
    layout vl{ov::PartialShape{1, 1, 2, 2}, data_types::f32, format::bfyx};
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, make_f14_topology(vl), config, false, false);
    ASSERT_NE(prog, nullptr);

    auto& rv = prog->get_node("rv");
    ASSERT_FALSE(rv.is_output());
    EXPECT_TRUE(rv.can_be_optimized());
}

TEST(inplace_alias_conflict_f12_f14, f14_read_value_output_not_optimized_dynamic) {
    auto& engine = get_test_engine();
    layout vl_dyn{ov::PartialShape{-1, 1, 2, 2}, data_types::f32, format::bfyx};
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"rv", "assign"}));
    auto prog = program::build_program(engine, make_f14_topology(vl_dyn), config, false, false);
    ASSERT_NE(prog, nullptr);

    auto& rv = prog->get_node("rv");
    ASSERT_TRUE(rv.is_output());
    EXPECT_FALSE(rv.can_be_optimized());
}
