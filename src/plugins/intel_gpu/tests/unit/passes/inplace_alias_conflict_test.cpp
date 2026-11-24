// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression tests for in-place / aliasing conflicts between optimized-out nodes (reorder->permute pair,
// no-op strided_slice) and in-place concatenation. Every test asserts the CORRECT numerical behavior.
// Tests that are still DISABLED_ document known, not-yet-fixed issues; run them with
//   --gtest_also_run_disabled_tests --gtest_filter=*inplace_alias_conflict*

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "openvino/op/interpolate.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/read_value.hpp"

#include "program_node.h"
#include "program_helpers.h"
#include "primitive_inst.h"

#include <iostream>
#include <sstream>

using namespace cldnn;
using namespace ::tests;

namespace {

std::vector<float> arange(size_t n, float start) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = start + static_cast<float>(i);
    return v;
}

std::string node_flags(const program& prog, const primitive_id& id) {
    std::ostringstream os;
    if (!prog.has_node(id)) {
        os << id << ": <not in program (removed)>";
        return os.str();
    }
    auto& n = prog.get_node(id);
    os << id << ": can_be_optimized=" << n.can_be_optimized() << " runtime_skippable=" << n.is_runtime_skippable()
       << " can_share_buffer=" << n.can_share_buffer() << " is_output=" << n.is_output()
       << " fmt=" << n.get_output_layout().format.to_string();
    return os.str();
}

void dump_flags(const program& prog, const std::vector<primitive_id>& ids) {
    for (auto& id : ids)
        std::cout << "[FLAGS] " << node_flags(prog, id) << std::endl;
}

std::vector<float> read_f32(const network_output& o, stream& s) {
    auto mem = o.get_memory();
    cldnn::mem_lock<float> p(mem, s);
    return std::vector<float>(p.begin(), p.end());
}

// bfyx logical concat of two tensors along `axis` (shapes b,f,y,x)
std::vector<float> host_concat(const std::vector<float>& a, const std::vector<size_t>& sa,
                               const std::vector<float>& b, const std::vector<size_t>& sb, size_t axis) {
    size_t outer = 1, inner_a = 1, inner_b = 1;
    for (size_t i = 0; i < axis; i++) outer *= sa[i];
    for (size_t i = axis; i < sa.size(); i++) inner_a *= sa[i];
    for (size_t i = axis; i < sb.size(); i++) inner_b *= sb[i];
    std::vector<float> r;
    for (size_t o = 0; o < outer; o++) {
        r.insert(r.end(), a.begin() + o * inner_a, a.begin() + (o + 1) * inner_a);
        r.insert(r.end(), b.begin() + o * inner_b, b.begin() + (o + 1) * inner_b);
    }
    return r;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Control: plain two-concat cascade, no crop. Must pass; proves the harness is sound.
// ---------------------------------------------------------------------------------------------
TEST(inplace_alias_conflict, control_concat_cascade_no_crop) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    layout la{ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx};
    layout lb{ov::PartialShape{1, 3, 2, 2}, data_types::f32, format::bfyx};
    layout ld{ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx};
    auto ma = engine.allocate_memory(la), mb = engine.allocate_memory(lb), md = engine.allocate_memory(ld);
    auto va = arange(8, 1), vb = arange(12, 101), vd = arange(8, 201);
    set_values(ma, va); set_values(mb, vb); set_values(md, vd);

    topology t;
    t.add(input_layout("A", la)); t.add(input_layout("B", lb)); t.add(input_layout("D", ld));
    t.add(activation("pa", input_info("A"), activation_func::relu));
    t.add(activation("pb", input_info("B"), activation_func::relu));
    t.add(activation("pd", input_info("D"), activation_func::relu));
    t.add(concatenation("c1", {input_info("pa"), input_info("pb")}, 1));
    t.add(concatenation("c2", {input_info("c1"), input_info("pd")}, 1));
    t.add(activation("out_c2", input_info("c2"), activation_func::relu));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, t, config, false, false);
    network net(prog, 0);
    net.set_input_data("A", ma); net.set_input_data("B", mb); net.set_input_data("D", md);
    auto outs = net.execute();
    dump_flags(*prog, {"pa", "pb", "pd", "c1", "c2", "out_c2"});

    auto ref = host_concat(host_concat(va, {1, 2, 2, 2}, vb, {1, 3, 2, 2}, 1), {1, 5, 2, 2}, vd, {1, 2, 2, 2}, 1);
    EXPECT_EQ(read_f32(outs.at("out_c2"), stream), ref);
}

// ---------------------------------------------------------------------------------------------
// Row 3: cascade concat C1 -> C2, and C1 also feeds a feature-axis crop with non-zero offset.
// ---------------------------------------------------------------------------------------------
TEST(inplace_alias_conflict, row3_cascade_concat_with_crop_on_inner_concat) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    layout la{ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx};
    layout lb{ov::PartialShape{1, 3, 2, 2}, data_types::f32, format::bfyx};
    layout ld{ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx};
    auto ma = engine.allocate_memory(la), mb = engine.allocate_memory(lb), md = engine.allocate_memory(ld);
    auto va = arange(8, 1), vb = arange(12, 101), vd = arange(8, 201);
    set_values(ma, va); set_values(mb, vb); set_values(md, vd);

    topology t;
    t.add(input_layout("A", la)); t.add(input_layout("B", lb)); t.add(input_layout("D", ld));
    t.add(activation("pa", input_info("A"), activation_func::relu));
    t.add(activation("pb", input_info("B"), activation_func::relu));
    t.add(activation("pd", input_info("D"), activation_func::relu));
    t.add(concatenation("c1", {input_info("pa"), input_info("pb")}, 1));
    t.add(concatenation("c2", {input_info("c1"), input_info("pd")}, 1));
    // crop features [1,4) of C1 (5 features): non-zero offset
    t.add(crop("crop", input_info("c1"), tensor(1, 3, 2, 2), tensor(0, 1, 0, 0)));
    t.add(activation("out_c2", input_info("c2"), activation_func::relu));
    t.add(activation("out_crop", input_info("crop"), activation_func::relu));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, t, config, false, false);
    network net(prog, 0);
    net.set_input_data("A", ma); net.set_input_data("B", mb); net.set_input_data("D", md);
    auto outs = net.execute();
    dump_flags(*prog, {"pa", "pb", "pd", "c1", "c2", "crop", "out_c2", "out_crop"});

    auto c2_mem = net.get_primitive("c2")->output_memory_ptr();
    auto c1_mem = net.get_primitive("c1")->output_memory_ptr();
    std::cout << "[FLAGS] c1 same_buffer c2 = " << engine.is_the_same_buffer(*c1_mem, *c2_mem) << std::endl;
    if (prog->has_node("crop")) {
        auto crop_mem = net.get_primitive("crop")->output_memory_ptr();
        std::cout << "[FLAGS] crop same_buffer c2 = " << engine.is_the_same_buffer(*crop_mem, *c2_mem)
                  << " crop same_buffer c1 = " << engine.is_the_same_buffer(*crop_mem, *c1_mem) << std::endl;
    }

    auto c1ref = host_concat(va, {1, 2, 2, 2}, vb, {1, 3, 2, 2}, 1);  // 5 features
    auto c2ref = host_concat(c1ref, {1, 5, 2, 2}, vd, {1, 2, 2, 2}, 1);
    std::vector<float> crop_ref(c1ref.begin() + 1 * 4, c1ref.begin() + 4 * 4);
    EXPECT_EQ(read_f32(outs.at("out_c2"), stream), c2ref);
    EXPECT_EQ(read_f32(outs.at("out_crop"), stream), crop_ref);
}

// ---------------------------------------------------------------------------------------------
// Row 5 control: same producer wired into a concat twice. Records result/flags only.
// ---------------------------------------------------------------------------------------------
TEST(inplace_alias_conflict, row5_same_producer_twice_into_concat) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    layout la{ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx};
    auto ma = engine.allocate_memory(la);
    auto va = arange(8, 1);
    set_values(ma, va);

    topology t;
    t.add(input_layout("A", la));
    t.add(activation("pa", input_info("A"), activation_func::relu));
    t.add(concatenation("c1", {input_info("pa"), input_info("pa")}, 1));
    t.add(activation("out_c1", input_info("c1"), activation_func::relu));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, t, config, false, false);
    network net(prog, 0);
    net.set_input_data("A", ma);
    auto outs = net.execute();
    dump_flags(*prog, {"pa", "c1", "out_c1"});
    EXPECT_EQ(read_f32(outs.at("out_c1"), stream), host_concat(va, {1, 2, 2, 2}, va, {1, 2, 2, 2}, 1));
}

// ---------------------------------------------------------------------------------------------
// Row 7: reorder -> identity permute -> concat predecessor.
// ---------------------------------------------------------------------------------------------
static void run_row7(format reorder_fmt, int concat_axis = 0) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    layout lx{ov::PartialShape{1, 16, 2, 3}, data_types::f32, format::bfyx};
    layout ly{ov::PartialShape{1, 16, 2, 3}, data_types::f32, format::bfyx};
    auto mx = engine.allocate_memory(lx), my = engine.allocate_memory(ly);
    auto vx = arange(96, 1), vy = arange(96, 1001);
    set_values(mx, vx); set_values(my, vy);

    topology t;
    t.add(input_layout("X", lx)); t.add(input_layout("Y", ly));
    t.add(reorder("rx", input_info("X"), reorder_fmt, data_types::f32));
    t.add(permute("px", input_info("rx"), {0, 1, 2, 3}));
    t.add(activation("py0", input_info("Y"), activation_func::relu));
    t.add(reorder("ry", input_info("py0"), reorder_fmt, data_types::f32));
    t.add(concatenation("cat", {input_info("px"), input_info("ry")}, concat_axis));
    t.add(reorder("out", input_info("cat"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, t, config, false, false);
    network net(prog, 0);
    net.set_input_data("X", mx); net.set_input_data("Y", my);
    auto outs = net.execute();
    dump_flags(*prog, {"rx", "px", "py0", "ry", "cat", "out"});
    if (prog->has_node("cat") && prog->has_node("px")) {
        auto cat_mem = net.get_primitive("cat")->output_memory_ptr();
        auto px_mem = net.get_primitive("px")->output_memory_ptr();
        auto rx_mem = prog->has_node("rx") ? net.get_primitive("rx")->output_memory_ptr() : nullptr;
        std::cout << "[FLAGS] px same_buffer cat = " << engine.is_the_same_buffer(*px_mem, *cat_mem);
        if (rx_mem)
            std::cout << " rx same_buffer cat = " << engine.is_the_same_buffer(*rx_mem, *cat_mem);
        std::cout << std::endl;
    }
    EXPECT_EQ(read_f32(outs.at("out"), stream), host_concat(vx, {1, 16, 2, 3}, vy, {1, 16, 2, 3}, static_cast<size_t>(concat_axis)));
}

TEST(inplace_alias_conflict, row7_reorder_identity_permute_concat_byxf) {
    run_row7(format::byxf);
}

TEST(inplace_alias_conflict, row7_reorder_identity_permute_concat_fsv16) {
    run_row7(format::b_fs_yx_fsv16);
}

// Row 7 controls, to attribute any wrong output.
//  mode 0: X -> reorder -> identity permute -> reorder(bfyx) -> output          (no concat at all)
//  mode 1: X -> reorder -> concat(with ry) -> reorder(bfyx)                      (no permute)
//  mode 2: X -> act -> permute(identity) -> concat(with ry)                      (permute pred, no reorder->permute pair)
static void run_row7_control(int mode, format fmt) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    layout lx{ov::PartialShape{1, 16, 2, 3}, data_types::f32, format::bfyx};
    auto mx = engine.allocate_memory(lx), my = engine.allocate_memory(lx);
    auto vx = arange(96, 1), vy = arange(96, 1001);
    set_values(mx, vx); set_values(my, vy);
    topology t;
    t.add(input_layout("X", lx)); t.add(input_layout("Y", lx));
    t.add(activation("py0", input_info("Y"), activation_func::relu));
    t.add(reorder("ry", input_info("py0"), fmt, data_types::f32));
    std::vector<float> ref;
    if (mode == 0) {
        t.add(reorder("rx", input_info("X"), fmt, data_types::f32));
        t.add(permute("px", input_info("rx"), {0, 1, 2, 3}));
        t.add(reorder("out", input_info("px"), format::bfyx, data_types::f32));
        ref = vx;
    } else {
        if (mode == 1) {
            t.add(reorder("px", input_info("X"), fmt, data_types::f32));
        } else {
            t.add(activation("pxa", input_info("X"), activation_func::relu));
            t.add(reorder("rx", input_info("pxa"), fmt, data_types::f32));
            t.add(activation("pxb", input_info("rx"), activation_func::relu));
            t.add(permute("px", input_info("pxb"), {0, 1, 2, 3}));
        }
        t.add(concatenation("cat", {input_info("px"), input_info("ry")}, 0));
        t.add(reorder("out", input_info("cat"), format::bfyx, data_types::f32));
        ref = host_concat(vx, {1, 16, 2, 3}, vy, {1, 16, 2, 3}, 0);
    }
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, t, config, false, false);
    network net(prog, 0);
    net.set_input_data("X", mx); net.set_input_data("Y", my);
    auto outs = net.execute();
    dump_flags(*prog, {"pxa", "rx", "pxb", "px", "py0", "ry", "cat", "out"});
    EXPECT_EQ(read_f32(outs.at("out"), stream), ref);
}

TEST(inplace_alias_conflict, row7_ctrl_reorder_permute_no_concat_byxf) { run_row7_control(0, format::byxf); }
TEST(inplace_alias_conflict, row7_ctrl_reorder_concat_no_permute_byxf) { run_row7_control(1, format::byxf); }
// Fails on the pre-F1 baseline (permute left can_be_optimized with byxf output, wrong permuted data) and passes with the fix.
// The activations around reorder/permute get fused away and remove_redundant_reorders later deletes the reorder, so the
// reorder->permute lambda's partner can vanish after it fired; the permute is then optimized without its reorder.
TEST(inplace_alias_conflict, row7_ctrl_act_permute_concat_byxf) { run_row7_control(2, format::byxf); }

TEST(inplace_alias_conflict, row7_reorder_identity_permute_concat_fsv16_axis1) {
    run_row7(format::b_fs_yx_fsv16, 1);
}

// ---------------------------------------------------------------------------------------------
// Row 8: no-op strided_slice as a concat predecessor.
// ---------------------------------------------------------------------------------------------
TEST(inplace_alias_conflict, row8_noop_strided_slice_concat_static) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    layout lx{ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx};
    layout ly{ov::PartialShape{1, 3, 3, 4}, data_types::f32, format::bfyx};
    auto mx = engine.allocate_memory(lx), my = engine.allocate_memory(ly);
    auto vx = arange(24, 1), vy = arange(36, 101);
    set_values(mx, vx); set_values(my, vy);

    topology t;
    t.add(input_layout("X", lx)); t.add(input_layout("Y", ly));
    t.add(activation("px", input_info("X"), activation_func::relu));
    t.add(strided_slice("ss", input_info("px"), {0, 0, 0, 0}, {1, 2, 3, 4}, {1, 1, 1, 1},
                        {}, {1, 1, 1, 1}, {}, {}, {}, ov::Shape{1, 2, 3, 4}));
    t.add(activation("py", input_info("Y"), activation_func::relu));
    t.add(concatenation("cat", {input_info("ss"), input_info("py")}, 1));
    t.add(activation("out", input_info("cat"), activation_func::relu));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, t, config, false, false);
    network net(prog, 0);
    net.set_input_data("X", mx); net.set_input_data("Y", my);
    auto outs = net.execute();
    dump_flags(*prog, {"px", "ss", "py", "cat", "out"});
    if (prog->has_node("ss") && prog->has_node("cat")) {
        auto cat_mem = net.get_primitive("cat")->output_memory_ptr();
        auto ss_mem = net.get_primitive("ss")->output_memory_ptr();
        auto px_mem = net.get_primitive("px")->output_memory_ptr();
        std::cout << "[FLAGS] ss same_buffer cat = " << engine.is_the_same_buffer(*ss_mem, *cat_mem)
                  << " px same_buffer cat = " << engine.is_the_same_buffer(*px_mem, *cat_mem)
                  << " ss same_buffer px = " << engine.is_the_same_buffer(*ss_mem, *px_mem) << std::endl;
    }
    EXPECT_EQ(read_f32(outs.at("out"), stream), host_concat(vx, {1, 2, 3, 4}, vy, {1, 3, 3, 4}, 1));
}

// F3 gates static nodes only. The non-empty end_mask makes the dynamic strided_slice reach the skippable-marking path;
// the dynamic behavior (flags, numerics) must be identical before and after the fix.
TEST(inplace_alias_conflict, row8_noop_strided_slice_concat_dynamic) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    layout lx_dyn{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    layout ly_dyn{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    layout lx{ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx};
    layout ly{ov::PartialShape{1, 3, 3, 4}, data_types::f32, format::bfyx};
    auto mx = engine.allocate_memory(lx), my = engine.allocate_memory(ly);
    auto vx = arange(24, 1), vy = arange(36, 101);
    set_values(mx, vx); set_values(my, vy);
    auto begin = engine.allocate_memory({ov::PartialShape{4}, data_types::i64, format::bfyx});
    auto end = engine.allocate_memory({ov::PartialShape{4}, data_types::i64, format::bfyx});
    auto strides = engine.allocate_memory({ov::PartialShape{4}, data_types::i64, format::bfyx});
    set_values<int64_t>(begin, {0, 0, 0, 0});
    set_values<int64_t>(end, {1, 2, 3, 4});
    set_values<int64_t>(strides, {1, 1, 1, 1});

    topology t;
    t.add(input_layout("X", lx_dyn)); t.add(input_layout("Y", ly_dyn));
    t.add(data("begin", begin)); t.add(data("end", end)); t.add(data("strides", strides));
    t.add(activation("px", input_info("X"), activation_func::relu));
    t.add(strided_slice("ss", input_info("px"), input_info("begin"), input_info("end"), input_info("strides"),
                        {}, {1, 1, 1, 1}, {}, {}, {}, {}));
    t.add(activation("py", input_info("Y"), activation_func::relu));
    t.add(concatenation("cat", {input_info("ss"), input_info("py")}, 1));
    t.add(activation("out", input_info("cat"), activation_func::relu));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, t, config, false, false);
    network net(prog, 0);
    net.set_input_data("X", mx); net.set_input_data("Y", my);
    auto outs = net.execute();
    dump_flags(*prog, {"px", "ss", "py", "cat", "out"});
    if (prog->has_node("ss")) {
        std::cout << "[FLAGS] dynamic ss is_dynamic=" << prog->get_node("ss").is_dynamic() << std::endl;
    }
    EXPECT_EQ(read_f32(outs.at("out"), stream), host_concat(vx, {1, 2, 3, 4}, vy, {1, 3, 3, 4}, 1));
}

// ---------------------------------------------------------------------------------------------
// Row 11: oneDNN eltwise-sum accumulated into a view whose owner has another reader.
//   conv1(in1) = X ; V = reshape(X) ; R = relu(X) (other reader of X's ORIGINAL data)
//   conv3(conv2b(conv2(in2))) with eltwise sum fused, sum-input V.
// ---------------------------------------------------------------------------------------------
static void run_row11(format conv_fmt, bool use_resample_view = false, bool late_reader = false) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    if (!engine.get_device_info().supports_immad) {
        std::cout << "[FLAGS] supports_immad=0 -> NOT RUN" << std::endl;
        GTEST_SKIP() << "oneDNN (immad) not available on this device";
    }
    layout l1{ov::PartialShape{1, 16, 32, 32}, data_types::f16, format::bfyx};
    layout l2{use_resample_view ? ov::PartialShape{1, 16, 32, 32} : ov::PartialShape{1, 16, 64, 16}, data_types::f16, format::bfyx};
    layout lw{ov::PartialShape{16, 16, 1, 1}, data_types::f16, format::bfyx};
    auto w = engine.allocate_memory(lw);
    {
        std::vector<ov::float16> wv(256);
        for (size_t o = 0; o < 16; o++)
            for (size_t i = 0; i < 16; i++)
                wv[o * 16 + i] = ov::float16((o == i) ? 0.5f : (((o + 2 * i) % 5 == 0) ? 0.125f : 0.f));
        set_values(w, wv);
    }
    auto in1 = engine.allocate_memory(l1), in2 = engine.allocate_memory(l2);
    {
        std::vector<ov::float16> v1(l1.count()), v2(l2.count());
        for (size_t i = 0; i < v1.size(); i++) v1[i] = ov::float16(static_cast<float>(i % 17) * 0.125f - 1.0f);
        for (size_t i = 0; i < v2.size(); i++) v2[i] = ov::float16(static_cast<float>((i * 3) % 13) * 0.0625f - 0.5f);
        set_values(in1, v1); set_values(in2, v2);
    }

    auto add_topo = [&](topology& t) {
        t.add(input_layout("in1", l1)); t.add(input_layout("in2", l2));
        t.add(data("w", w));
        t.add(convolution("conv1", input_info("in1"), "w", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        if (use_resample_view)
            t.add(resample("view", input_info("conv1"), tensor(1, 16, 32, 32), 1, ov::op::v4::Interpolate::InterpolateMode::NEAREST));
        else
            t.add(reshape("view", input_info("conv1"), false, std::vector<int64_t>{1, 16, 64, 16}, ov::PartialShape{1, 16, 64, 16}));
        t.add(activation("reader", input_info("conv1"), activation_func::abs));
        t.add(reorder("out_reader", input_info("reader"), format::bfyx, data_types::f32));
        t.add(convolution("conv2", input_info("in2"), "w", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        t.add(convolution("conv2b", input_info("conv2"), "w", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        t.add(convolution("conv3", input_info("conv2b"), "w", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        t.add(eltwise("sum", input_info("conv3"), input_info("view"), eltwise_mode::sum));
        t.add(reorder("out_sum", input_info("sum"), format::bfyx, data_types::f32));
        if (late_reader) {
            // reads X (conv1) AFTER the sum result exists: topologically forced to run after conv3
            t.add(eltwise("late", input_info("conv1"), input_info("sum"), eltwise_mode::prod));
            t.add(reorder("out_late", input_info("late"), format::bfyx, data_types::f32));
        }
    };
    topology test_t, ref_t;
    add_topo(test_t); add_topo(ref_t);

    ov::intel_gpu::ImplementationDesc od = {conv_fmt, "", impl_types::onednn};
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplForcingMap forced{{"conv1", od}, {"conv2", od}, {"conv2b", od}, {"conv3", od}};
    if (use_resample_view)
        forced.emplace("view", ov::intel_gpu::ImplementationDesc{conv_fmt, "", impl_types::ocl});
    config.set_property(ov::intel_gpu::force_implementations(forced));
    ExecutionConfig config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::optimize_data(false));

    network net_ref(engine, ref_t, config_ref);
    net_ref.set_input_data("in1", in1); net_ref.set_input_data("in2", in2);
    auto outs_ref = net_ref.execute();

    auto prog = program::build_program(engine, test_t, config);
    network net(prog, 0);
    net.set_input_data("in1", in1); net.set_input_data("in2", in2);
    auto outs = net.execute();
    dump_flags(*prog, {"conv1", "view", "reader", "conv2", "conv2b", "conv3", "sum", "late"});
    if (prog->has_node("conv3")) {
        auto& c3 = prog->get_node("conv3");
        for (const auto& fd : c3.get_fused_primitives()) {
            if (!fd.is_type<eltwise>() || !fd.has_outer_dep()) continue;
            auto ft = onednn_eltwise_fusing_helpers::get_add_fusing_type(c3, fd);
            std::cout << "[FLAGS] conv3 fused eltwise add_fusing_type=" << static_cast<int>(ft)
                      << " (sum=" << static_cast<int>(onednn_eltwise_fusing_helpers::add_fusing_type::sum) << ")"
                      << " outer_dep=" << c3.get_dependencies()[fd.outer_dep_start_idx].first->id() << std::endl;
        }
    }

    EXPECT_EQ(read_f32(outs.at("out_reader"), stream), read_f32(outs_ref.at("out_reader"), stream));
    EXPECT_EQ(read_f32(outs.at("out_sum"), stream), read_f32(outs_ref.at("out_sum"), stream));
    if (late_reader) {
        EXPECT_EQ(read_f32(outs.at("out_late"), stream), read_f32(outs_ref.at("out_late"), stream));
    }
}

// Row 11 tests stay disabled: the oneDNN sum-into-view aliasing issue is not fixed by this change.
TEST(inplace_alias_conflict, DISABLED_row11_onednn_sum_into_view_with_other_reader_fsv16) {
    run_row11(format::b_fs_yx_fsv16);
}

TEST(inplace_alias_conflict, DISABLED_row11_onednn_sum_into_view_with_other_reader_bfyx) {
    run_row11(format::bfyx);
}

// Same as above but the "view" is an optimized-out NEAREST resample of identical size (the pattern of the existing
// optimized_resample_to_onednn_sum_reuse_correctness test) and conv1 (the owner) gets an additional reader.
TEST(inplace_alias_conflict, DISABLED_row11_onednn_sum_into_resample_view_with_other_reader) {
    run_row11(format::b_fs_yx_fsv16, true);
}

// As above, plus a second reader of the owner conv1 that is forced (data dependency on the sum result) to run AFTER conv3.
TEST(inplace_alias_conflict, DISABLED_row11_onednn_sum_into_resample_view_with_late_reader) {
    run_row11(format::b_fs_yx_fsv16, true, true);
}

// ---------------------------------------------------------------------------------------------
// F29 (candidate, Phase 0.5b design 13 question 10; fixed in Phase 1a): the oneDNN sum overwrites A, A has two users {R, C-chain} where R
// is a cbo view of A, and R has another reader Y that runs AFTER the sum. F11 only looks at the dependency of the sum writer
// (an owner other than A), F10/F6 do not fire, hence the sum is formed and Y reads overwritten data.
//   A = conv1(in1); R = resample(A) (cbo view, same size); X = conv2(R); C = conv3(X) with sum(A) fused; Y = prod(R, relu(C)).
// N10: the test first asserts that the sum is really formed; otherwise it reports "construction not formed" with GTEST_SKIP.
// ---------------------------------------------------------------------------------------------
TEST(inplace_alias_conflict, DISABLED_row11b_onednn_sum_into_owner_with_late_reader_of_ancestor_view) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN (immad) not available on this device";
    }
    layout l1{ov::PartialShape{1, 16, 32, 32}, data_types::f16, format::bfyx};
    layout lw{ov::PartialShape{16, 16, 1, 1}, data_types::f16, format::bfyx};
    auto w = engine.allocate_memory(lw);
    {
        std::vector<ov::float16> wv(256);
        for (size_t o = 0; o < 16; o++)
            for (size_t i = 0; i < 16; i++)
                wv[o * 16 + i] = ov::float16((o == i) ? 0.5f : (((o + 2 * i) % 5 == 0) ? 0.125f : 0.f));
        set_values(w, wv);
    }
    auto in1 = engine.allocate_memory(l1);
    {
        std::vector<ov::float16> v1(l1.count());
        for (size_t i = 0; i < v1.size(); i++) v1[i] = ov::float16(static_cast<float>(i % 17) * 0.125f - 1.0f);
        set_values(in1, v1);
    }
    auto add_topo = [&](topology& t) {
        t.add(input_layout("in1", l1));
        t.add(data("w", w));
        t.add(convolution("A", input_info("in1"), "w", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        t.add(resample("R", input_info("A"), tensor(1, 16, 32, 32), 1, ov::op::v4::Interpolate::InterpolateMode::NEAREST));
        t.add(convolution("X", input_info("R"), "w", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        t.add(convolution("Cc", input_info("X"), "w", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        t.add(eltwise("C", input_info("Cc"), input_info("A"), eltwise_mode::sum));
        t.add(activation("fC", input_info("C"), activation_func::relu));
        t.add(eltwise("Y", input_info("R"), input_info("fC"), eltwise_mode::prod));
        t.add(reorder("out_y", input_info("Y"), format::bfyx, data_types::f32));
        t.add(reorder("out_c", input_info("C"), format::bfyx, data_types::f32));
    };
    topology test_t, ref_t;
    add_topo(test_t);
    add_topo(ref_t);
    ov::intel_gpu::ImplementationDesc od = {format::b_fs_yx_fsv16, "", impl_types::onednn};
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"A", od}, {"X", od}, {"Cc", od}, {"R", ov::intel_gpu::ImplementationDesc{format::b_fs_yx_fsv16, "", impl_types::ocl}}}));
    ExecutionConfig config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::optimize_data(false));

    network net_ref(engine, ref_t, config_ref);
    net_ref.set_input_data("in1", in1);
    auto outs_ref = net_ref.execute();

    auto prog = program::build_program(engine, test_t, config);
    // N10: the construction must really contain the sum into A (outer dependency A) and the cbo view R
    bool sum_formed = false;
    if (prog->has_node("Cc") && prog->has_node("A") && prog->has_node("R")) {
        auto& c3 = prog->get_node("Cc");
        for (const auto& fd : c3.get_fused_primitives()) {
            if (fd.is_type<eltwise>() && fd.has_outer_dep()) {
                sum_formed = onednn_eltwise_fusing_helpers::get_add_fusing_type(c3, fd) == onednn_eltwise_fusing_helpers::add_fusing_type::sum &&
                             c3.get_dependencies()[fd.outer_dep_start_idx].first->id() == "A";
            }
        }
        sum_formed = sum_formed && prog->get_node("R").can_be_optimized();
    }
    if (!sum_formed) {
        dump_flags(*prog, {"A", "R", "X", "Cc", "C", "fC", "Y"});
        GTEST_SKIP() << "construction not formed: the oneDNN sum into A with the cbo view R was not created by the passes";
    }
    network net(prog, 0);
    net.set_input_data("in1", in1);
    auto outs = net.execute();
    // oneDNN fp16 vs the unoptimized OCL reference: a precision difference is not the defect, so compare with a tolerance (abs 2e-2 plus 5% of
    // the reference). The wrong data of the late reader differs by far more (observed: 0.0586 vs 0.0151, 0.0020 vs -0.0023 sign flips).
    auto expect_close = [&](const std::string& id) {
        const auto got = read_f32(outs.at(id), stream);
        const auto ref = read_f32(outs_ref.at(id), stream);
        ASSERT_EQ(got.size(), ref.size()) << id;
        size_t bad = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (std::abs(got[i] - ref[i]) > 2e-2f + 0.05f * std::abs(ref[i])) {
                ++bad;
            }
        }
        EXPECT_EQ(bad, 0u) << id << ": " << bad << " of " << got.size() << " elements differ beyond the fp16 tolerance";
    };
    expect_close("out_y");
    expect_close("out_c");
}

// ---------------------------------------------------------------------------------------------
// read_value / assign ordering (candidate, Phase 0.5b design 13 section 10, folded into the Phase 1a read_value/assign entry): an optimized
// read_value returns the variable buffer itself. can_read_value_be_optimize (prepare_buffer_fusing.cpp ~:463-464) accepts a read_value with
// a single reader without looking where that reader runs relative to the assign: a reader that runs AFTER the assign sees the new value.
// Expected (not true today): a read_value whose only reader is ordered after an assign of the same variable is not optimized.
// ---------------------------------------------------------------------------------------------
TEST(inplace_alias_conflict, DISABLED_optimized_read_value_single_reader_after_assign) {
    auto& engine = get_test_engine();
    layout l{ov::PartialShape{1, 4, 8, 8}, data_types::f32, format::bfyx};
    topology t;
    t.add(input_layout("in", l));
    t.add(input_layout("in2", l));
    t.add(read_value{"rv", {input_info("in")}, "v0", {l}});
    t.add(activation("src", input_info("in2"), activation_func::relu));
    t.add(assign("as", {input_info("src")}, "v0", l));
    // the reader is the only user of rv and depends on the assign output too, so it is ordered after the assign
    t.add(eltwise("late_reader", {input_info("rv"), input_info("as")}, eltwise_mode::sum));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    program::ptr prog;
    try {
        prog = program::build_program(engine, t, config, false, false);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "construction not formed: graph cannot be built here: " << e.what();
    }
    if (!prog->has_node("rv") || !prog->has_node("as") || !prog->has_node("late_reader")) {
        std::string ids;
        for (auto* n : prog->get_processing_order()) {
            ids += n->id() + " ";
        }
        GTEST_SKIP() << "construction not formed: nodes were removed [" << ids << "] (rv=" << prog->has_node("rv") << " as=" << prog->has_node("as")
                     << " late_reader=" << prog->has_node("late_reader") << ")";
    }
    size_t pos_as = 0, pos_reader = 0, i = 0;
    for (auto* n : prog->get_processing_order()) {
        pos_as = n->id() == "as" ? i : pos_as;
        pos_reader = n->id() == "late_reader" ? i : pos_reader;
        ++i;
    }
    if (!(pos_as < pos_reader) || prog->get_node("rv").get_users().size() != 1) {
        GTEST_SKIP() << "construction not formed: the reader is not the only user of rv or does not run after the assign";
    }
    EXPECT_FALSE(prog->get_node("rv").can_be_optimized()) << "read_value optimized although its only reader runs after the assign";
}

// ---------------------------------------------------------------------------------------------
// Legitimate reorder->permute pair: byxf source, reorder to bfyx, permute {0,2,3,1}. The pair is a physical no-op and
// must still be optimized out, with correct numerical result.
// ---------------------------------------------------------------------------------------------
TEST(inplace_alias_conflict, reorder_permute_pair_byxf_source_still_optimized) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    layout lx{ov::PartialShape{1, 16, 2, 3}, data_types::f32, format::byxf};
    auto mx = engine.allocate_memory(lx);
    auto vx = arange(96, 1);  // physical byxf order
    set_values(mx, vx);

    topology t;
    t.add(input_layout("X", lx));
    t.add(reorder("rx", input_info("X"), format::bfyx, data_types::f32));
    t.add(permute("px", input_info("rx"), {0, 2, 3, 1}));
    t.add(activation("out", input_info("px"), activation_func::relu));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, t, config, false, false);
    ASSERT_TRUE(prog->has_node("rx"));
    ASSERT_TRUE(prog->has_node("px"));
    EXPECT_TRUE(prog->get_node("rx").can_be_optimized());
    EXPECT_TRUE(prog->get_node("px").can_be_optimized());

    network net(prog, 0);
    net.set_input_data("X", mx);
    auto outs = net.execute();
    // permuted logical shape is (b, y, x, f) in bfyx whose bytes equal the byxf source bytes
    EXPECT_EQ(read_f32(outs.at("out"), stream), vx);
}

// ---------------------------------------------------------------------------------------------
// F2 standalone: fyxb source -> reorder(bfyx) -> permute {1,2,3,0} (the dims order of the source). F1 passes (reorder and
// permute outputs are both bfyx) and the permute is not "rotating except batch", so it is an admissible in-place concat
// predecessor. The concat plans for permute to write into the concat buffer, hence the pair must NOT be optimized out (F2).
// ---------------------------------------------------------------------------------------------
TEST(inplace_alias_conflict, reorder_permute_pair_into_inplace_concat_not_optimized) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    layout lx{ov::PartialShape{1, 16, 2, 3}, data_types::f32, format::fyxb};
    layout ly{ov::PartialShape{16, 2, 3, 1}, data_types::f32, format::bfyx};
    auto mx = engine.allocate_memory(lx), my = engine.allocate_memory(ly);
    auto vx = arange(96, 1), vy = arange(96, 1001);  // vx in physical fyxb order
    set_values(mx, vx); set_values(my, vy);

    topology t;
    t.add(input_layout("X", lx)); t.add(input_layout("Y", ly));
    t.add(reorder("rx", input_info("X"), format::bfyx, data_types::f32));
    t.add(permute("px", input_info("rx"), {1, 2, 3, 0}));
    t.add(activation("py", input_info("Y"), activation_func::relu));
    t.add(concatenation("cat", {input_info("px"), input_info("py")}, 1));
    t.add(activation("out", input_info("cat"), activation_func::relu));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, t, config, false, false);
    dump_flags(*prog, {"rx", "px", "py", "cat", "out"});
    ASSERT_TRUE(prog->has_node("px"));
    ASSERT_TRUE(prog->has_node("cat"));
    EXPECT_TRUE(prog->get_node("cat").can_be_optimized());
    EXPECT_FALSE(prog->get_node("px").can_be_optimized());

    network net(prog, 0);
    net.set_input_data("X", mx); net.set_input_data("Y", my);
    auto outs = net.execute();
    // permuted logical (f,y,x,b) in bfyx has the same bytes as the fyxb source
    EXPECT_EQ(read_f32(outs.at("out"), stream), host_concat(vx, {16, 2, 3, 1}, vy, {16, 2, 3, 1}, 1));
}
