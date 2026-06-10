// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/internal_properties.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/pooling.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/reduce.hpp>
#include <intel_gpu/primitives/deconvolution.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "openvino/core/type/bfloat16.hpp"

#include <cmath>
#include <vector>

using namespace cldnn;
using namespace ::tests;

#ifdef ENABLE_ONEDNN_FOR_GPU

// Helper: generate random f32 values and convert to bf16
static std::vector<ov::bfloat16> generate_bf16_data(tests::random_generator& rg, size_t count) {
    auto f32_data = rg.generate_random_1d<float>(count, -1.0f, 1.0f);
    std::vector<ov::bfloat16> bf16_data(count);
    for (size_t i = 0; i < count; ++i) {
        bf16_data[i] = ov::bfloat16(f32_data[i]);
    }
    return bf16_data;
}

// Helper: compare f32 outputs with tolerance
static void compare_outputs(const std::vector<float>& ref, const std::vector<float>& actual, float tolerance) {
    ASSERT_EQ(ref.size(), actual.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i]) < 1e-5f) {
            ASSERT_NEAR(actual[i], ref[i], tolerance)
                << "Mismatch at index " << i;
        } else {
            float rel_err = std::abs(actual[i] - ref[i]) / std::max(std::abs(ref[i]), 1e-5f);
            ASSERT_LT(rel_err, tolerance)
                << "Relative error at index " << i << ": " << rel_err
                << " (ref=" << ref[i] << ", actual=" << actual[i] << ")";
        }
    }
}

// Helper: run a network and get output as f32 vector
// If force_onednn_for is non-empty, forces the given primitive to use oneDNN implementation.
static std::vector<float> run_network_get_f32_output(engine& eng, topology& tp,
                                                      const std::string& input_name, memory::ptr input_mem,
                                                      const std::string& output_name,
                                                      const std::string& force_onednn_for = "") {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    if (!force_onednn_for.empty()) {
        ov::intel_gpu::ImplementationDesc onednn_impl = { format::bfyx, "", impl_types::onednn };
        config.set_property(ov::intel_gpu::force_implementations(
            ov::intel_gpu::ImplForcingMap{ {force_onednn_for, onednn_impl} }));
    }
    network net(eng, tp, config);
    net.set_input_data(input_name, input_mem);
    auto outputs = net.execute();
    auto out_mem = outputs.at(output_name).get_memory();
    mem_lock<float> out_ptr(out_mem, get_test_stream());
    return std::vector<float>(out_ptr.begin(), out_ptr.end());
}

// =====================================================
// TEST: Convolution bf16
// =====================================================
TEST(bf16_onednn_ops, convolution_bf16) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "BF16 oneDNN requires XMX/DPAS support (Xe-HPG+)";

    tests::random_generator rg("bf16_conv");
    const int batch = 1, in_f = 16, in_y = 8, in_x = 8;
    const int out_f = 16, k_y = 3, k_x = 3;

    auto input_bf16 = generate_bf16_data(rg, batch * in_f * in_y * in_x);
    auto weights_bf16 = generate_bf16_data(rg, out_f * in_f * k_y * k_x);

    // Convert bf16 data to f32 for reference
    std::vector<float> input_f32(input_bf16.size());
    std::vector<float> weights_f32(weights_bf16.size());
    for (size_t i = 0; i < input_bf16.size(); ++i) input_f32[i] = static_cast<float>(input_bf16[i]);
    for (size_t i = 0; i < weights_bf16.size(); ++i) weights_f32[i] = static_cast<float>(weights_bf16[i]);

    // BF16 path
    auto input_mem_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { batch, in_f, in_x, in_y } });
    auto weights_mem_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { out_f, in_f, k_x, k_y } });
    set_values(input_mem_bf16, input_bf16);
    set_values(weights_mem_bf16, weights_bf16);

    topology tp_bf16;
    tp_bf16.add(input_layout("input", input_mem_bf16->get_layout()));
    tp_bf16.add(data("weights", weights_mem_bf16));
    tp_bf16.add(convolution("conv", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
    tp_bf16.add(reorder("output", input_info("conv"), format::bfyx, data_types::f32));

    auto result_bf16 = run_network_get_f32_output(engine, tp_bf16, "input", input_mem_bf16, "output", "conv");

    // F32 reference path
    auto input_mem_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, in_f, in_x, in_y } });
    auto weights_mem_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { out_f, in_f, k_x, k_y } });
    set_values(input_mem_f32, input_f32);
    set_values(weights_mem_f32, weights_f32);

    topology tp_f32;
    tp_f32.add(input_layout("input", input_mem_f32->get_layout()));
    tp_f32.add(data("weights", weights_mem_f32));
    tp_f32.add(convolution("conv", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
    tp_f32.add(reorder("output", input_info("conv"), format::bfyx, data_types::f32));

    auto result_f32 = run_network_get_f32_output(engine, tp_f32, "input", input_mem_f32, "output");

    // bf16 has 7-bit mantissa → ~0.8% relative error per multiply-accumulate
    // With 16 input channels and 3x3 kernel = 144 MACs, error can accumulate
    compare_outputs(result_f32, result_bf16, 0.05f);  // 5% tolerance
}

// =====================================================
// TEST: Gemm bf16
// =====================================================
TEST(bf16_onednn_ops, gemm_bf16) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "BF16 oneDNN requires XMX/DPAS support (Xe-HPG+)";

    tests::random_generator rg("bf16_gemm");

    const int M = 16, K = 32, N = 16;

    auto input0_bf16 = generate_bf16_data(rg, M * K);
    auto input1_bf16 = generate_bf16_data(rg, K * N);

    std::vector<float> input0_f32(input0_bf16.size());
    std::vector<float> input1_f32(input1_bf16.size());
    for (size_t i = 0; i < input0_bf16.size(); ++i) input0_f32[i] = static_cast<float>(input0_bf16[i]);
    for (size_t i = 0; i < input1_bf16.size(); ++i) input1_f32[i] = static_cast<float>(input1_bf16[i]);

    // BF16 path
    auto mem0_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { 1, 1, K, M } });
    auto mem1_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { 1, 1, N, K } });
    set_values(mem0_bf16, input0_bf16);
    set_values(mem1_bf16, input1_bf16);

    topology tp_bf16;
    tp_bf16.add(input_layout("input0", mem0_bf16->get_layout()));
    tp_bf16.add(input_layout("input1", mem1_bf16->get_layout()));
    tp_bf16.add(gemm("gemm", { input_info("input0"), input_info("input1") }, data_types::bf16, false, false));
    tp_bf16.add(reorder("output", input_info("gemm"), format::bfyx, data_types::f32));

    ExecutionConfig config_bf16 = get_test_default_config(engine);
    config_bf16.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc gemm_onednn_impl = { format::bfyx, "", impl_types::onednn };
    config_bf16.set_property(ov::intel_gpu::force_implementations(
        ov::intel_gpu::ImplForcingMap{ {"gemm", gemm_onednn_impl} }));
    network net_bf16(engine, tp_bf16, config_bf16);
    net_bf16.set_input_data("input0", mem0_bf16);
    net_bf16.set_input_data("input1", mem1_bf16);
    auto outputs_bf16 = net_bf16.execute();
    auto out_bf16 = outputs_bf16.at("output").get_memory();
    mem_lock<float> result_bf16_ptr(out_bf16, get_test_stream());
    std::vector<float> result_bf16(result_bf16_ptr.begin(), result_bf16_ptr.end());

    // F32 reference path
    auto mem0_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, K, M } });
    auto mem1_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, N, K } });
    set_values(mem0_f32, input0_f32);
    set_values(mem1_f32, input1_f32);

    topology tp_f32;
    tp_f32.add(input_layout("input0", mem0_f32->get_layout()));
    tp_f32.add(input_layout("input1", mem1_f32->get_layout()));
    tp_f32.add(gemm("gemm", { input_info("input0"), input_info("input1") }, data_types::f32, false, false));
    tp_f32.add(reorder("output", input_info("gemm"), format::bfyx, data_types::f32));

    ExecutionConfig config_f32 = get_test_default_config(engine);
    config_f32.set_property(ov::intel_gpu::optimize_data(true));
    network net_f32(engine, tp_f32, config_f32);
    net_f32.set_input_data("input0", mem0_f32);
    net_f32.set_input_data("input1", mem1_f32);
    auto outputs_f32 = net_f32.execute();
    auto out_f32 = outputs_f32.at("output").get_memory();
    mem_lock<float> result_f32_ptr(out_f32, get_test_stream());
    std::vector<float> result_f32(result_f32_ptr.begin(), result_f32_ptr.end());

    compare_outputs(result_f32, result_bf16, 0.05f);
}

// =====================================================
// TEST: Fully Connected bf16
// =====================================================
TEST(bf16_onednn_ops, fully_connected_bf16) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "BF16 oneDNN requires XMX/DPAS support (Xe-HPG+)";

    tests::random_generator rg("bf16_fc");

    const int batch = 4, input_f = 64, output_f = 32;

    auto input_bf16 = generate_bf16_data(rg, batch * input_f);
    auto weights_bf16 = generate_bf16_data(rg, output_f * input_f);

    std::vector<float> input_f32(input_bf16.size());
    std::vector<float> weights_f32(weights_bf16.size());
    for (size_t i = 0; i < input_bf16.size(); ++i) input_f32[i] = static_cast<float>(input_bf16[i]);
    for (size_t i = 0; i < weights_bf16.size(); ++i) weights_f32[i] = static_cast<float>(weights_bf16[i]);

    // BF16 path
    auto input_mem_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { batch, input_f, 1, 1 } });
    auto weights_mem_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { output_f, input_f, 1, 1 } });
    set_values(input_mem_bf16, input_bf16);
    set_values(weights_mem_bf16, weights_bf16);

    topology tp_bf16;
    tp_bf16.add(input_layout("input", input_mem_bf16->get_layout()));
    tp_bf16.add(data("weights", weights_mem_bf16));
    tp_bf16.add(fully_connected("fc", input_info("input"), "weights"));
    tp_bf16.add(reorder("output", input_info("fc"), format::bfyx, data_types::f32));

    auto result_bf16 = run_network_get_f32_output(engine, tp_bf16, "input", input_mem_bf16, "output", "fc");

    // F32 reference path
    auto input_mem_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, input_f, 1, 1 } });
    auto weights_mem_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { output_f, input_f, 1, 1 } });
    set_values(input_mem_f32, input_f32);
    set_values(weights_mem_f32, weights_f32);

    topology tp_f32;
    tp_f32.add(input_layout("input", input_mem_f32->get_layout()));
    tp_f32.add(data("weights", weights_mem_f32));
    tp_f32.add(fully_connected("fc", input_info("input"), "weights"));
    tp_f32.add(reorder("output", input_info("fc"), format::bfyx, data_types::f32));

    auto result_f32 = run_network_get_f32_output(engine, tp_f32, "input", input_mem_f32, "output");

    compare_outputs(result_f32, result_bf16, 0.05f);
}

// =====================================================
// TEST: Pooling (avg) bf16
// =====================================================
TEST(bf16_onednn_ops, pooling_avg_bf16) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "BF16 oneDNN requires XMX/DPAS support (Xe-HPG+)";

    tests::random_generator rg("bf16_pool");

    const int batch = 1, features = 16, in_y = 8, in_x = 8;
    auto input_bf16 = generate_bf16_data(rg, batch * features * in_y * in_x);

    std::vector<float> input_f32(input_bf16.size());
    for (size_t i = 0; i < input_bf16.size(); ++i) input_f32[i] = static_cast<float>(input_bf16[i]);

    // BF16 path
    auto input_mem_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { batch, features, in_x, in_y } });
    set_values(input_mem_bf16, input_bf16);

    topology tp_bf16;
    tp_bf16.add(input_layout("input", input_mem_bf16->get_layout()));
    tp_bf16.add(pooling("pool", input_info("input"), pooling_mode::average, { 2, 2 }, { 2, 2 }));
    tp_bf16.add(reorder("output", input_info("pool"), format::bfyx, data_types::f32));

    auto result_bf16 = run_network_get_f32_output(engine, tp_bf16, "input", input_mem_bf16, "output", "pool");

    // F32 reference path
    auto input_mem_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, features, in_x, in_y } });
    set_values(input_mem_f32, input_f32);

    topology tp_f32;
    tp_f32.add(input_layout("input", input_mem_f32->get_layout()));
    tp_f32.add(pooling("pool", input_info("input"), pooling_mode::average, { 2, 2 }, { 2, 2 }));
    tp_f32.add(reorder("output", input_info("pool"), format::bfyx, data_types::f32));

    auto result_f32 = run_network_get_f32_output(engine, tp_f32, "input", input_mem_f32, "output");

    compare_outputs(result_f32, result_bf16, 0.02f);  // pooling is simple avg, low error
}

// =====================================================
// TEST: Concatenation bf16
// =====================================================
TEST(bf16_onednn_ops, concatenation_bf16) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "BF16 oneDNN requires XMX/DPAS support (Xe-HPG+)";

    tests::random_generator rg("bf16_concat");

    const int batch = 1, f1 = 8, f2 = 8, y = 4, x = 4;
    auto data1_bf16 = generate_bf16_data(rg, batch * f1 * y * x);
    auto data2_bf16 = generate_bf16_data(rg, batch * f2 * y * x);

    std::vector<float> data1_f32(data1_bf16.size()), data2_f32(data2_bf16.size());
    for (size_t i = 0; i < data1_bf16.size(); ++i) data1_f32[i] = static_cast<float>(data1_bf16[i]);
    for (size_t i = 0; i < data2_bf16.size(); ++i) data2_f32[i] = static_cast<float>(data2_bf16[i]);

    // BF16 path
    auto mem1_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { batch, f1, x, y } });
    auto mem2_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { batch, f2, x, y } });
    set_values(mem1_bf16, data1_bf16);
    set_values(mem2_bf16, data2_bf16);

    topology tp_bf16;
    tp_bf16.add(input_layout("in1", mem1_bf16->get_layout()));
    tp_bf16.add(input_layout("in2", mem2_bf16->get_layout()));
    tp_bf16.add(concatenation("concat", { input_info("in1"), input_info("in2") }, 1));
    tp_bf16.add(reorder("output", input_info("concat"), format::bfyx, data_types::f32));

    ExecutionConfig config_bf16 = get_test_default_config(engine);
    config_bf16.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc concat_onednn_impl = { format::bfyx, "", impl_types::onednn };
    config_bf16.set_property(ov::intel_gpu::force_implementations(
        ov::intel_gpu::ImplForcingMap{ {"concat", concat_onednn_impl} }));
    network net_bf16(engine, tp_bf16, config_bf16);
    net_bf16.set_input_data("in1", mem1_bf16);
    net_bf16.set_input_data("in2", mem2_bf16);
    auto outputs_bf16 = net_bf16.execute();
    auto out_bf16 = outputs_bf16.at("output").get_memory();
    mem_lock<float> result_bf16_ptr(out_bf16, get_test_stream());
    std::vector<float> result_bf16(result_bf16_ptr.begin(), result_bf16_ptr.end());

    // F32 reference
    auto mem1_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, f1, x, y } });
    auto mem2_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, f2, x, y } });
    set_values(mem1_f32, data1_f32);
    set_values(mem2_f32, data2_f32);

    topology tp_f32;
    tp_f32.add(input_layout("in1", mem1_f32->get_layout()));
    tp_f32.add(input_layout("in2", mem2_f32->get_layout()));
    tp_f32.add(concatenation("concat", { input_info("in1"), input_info("in2") }, 1));
    tp_f32.add(reorder("output", input_info("concat"), format::bfyx, data_types::f32));

    ExecutionConfig config_f32 = get_test_default_config(engine);
    config_f32.set_property(ov::intel_gpu::optimize_data(true));
    network net_f32(engine, tp_f32, config_f32);
    net_f32.set_input_data("in1", mem1_f32);
    net_f32.set_input_data("in2", mem2_f32);
    auto outputs_f32 = net_f32.execute();
    auto out_f32 = outputs_f32.at("output").get_memory();
    mem_lock<float> result_f32_ptr(out_f32, get_test_stream());
    std::vector<float> result_f32(result_f32_ptr.begin(), result_f32_ptr.end());

    // Concatenation is just data movement, should be exact match
    compare_outputs(result_f32, result_bf16, 0.001f);
}

// =====================================================
// TEST: Reduce (sum) bf16
// =====================================================
TEST(bf16_onednn_ops, reduce_sum_bf16) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "BF16 oneDNN requires XMX/DPAS support (Xe-HPG+)";

    tests::random_generator rg("bf16_reduce");

    const int batch = 1, features = 16, in_y = 4, in_x = 4;
    auto input_bf16 = generate_bf16_data(rg, batch * features * in_y * in_x);

    std::vector<float> input_f32(input_bf16.size());
    for (size_t i = 0; i < input_bf16.size(); ++i) input_f32[i] = static_cast<float>(input_bf16[i]);

    // BF16 path
    auto input_mem_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { batch, features, in_x, in_y } });
    set_values(input_mem_bf16, input_bf16);

    topology tp_bf16;
    tp_bf16.add(input_layout("input", input_mem_bf16->get_layout()));
    tp_bf16.add(reduce("reduce", input_info("input"), reduce_mode::sum, {2, 3}, true));
    tp_bf16.add(reorder("output", input_info("reduce"), format::bfyx, data_types::f32));

    auto result_bf16 = run_network_get_f32_output(engine, tp_bf16, "input", input_mem_bf16, "output", "reduce");

    // F32 reference
    auto input_mem_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, features, in_x, in_y } });
    set_values(input_mem_f32, input_f32);

    topology tp_f32;
    tp_f32.add(input_layout("input", input_mem_f32->get_layout()));
    tp_f32.add(reduce("reduce", input_info("input"), reduce_mode::sum, {2, 3}, true));
    tp_f32.add(reorder("output", input_info("reduce"), format::bfyx, data_types::f32));

    auto result_f32 = run_network_get_f32_output(engine, tp_f32, "input", input_mem_f32, "output");

    compare_outputs(result_f32, result_bf16, 0.05f);
}

// =====================================================
// TEST: Deconvolution bf16
// =====================================================
TEST(bf16_onednn_ops, deconvolution_bf16) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "BF16 oneDNN requires XMX/DPAS support (Xe-HPG+)";

    tests::random_generator rg("bf16_deconv");

    const int batch = 1, in_f = 16, in_y = 4, in_x = 4;
    const int out_f = 16, k_y = 3, k_x = 3;

    auto input_bf16 = generate_bf16_data(rg, batch * in_f * in_y * in_x);
    auto weights_bf16 = generate_bf16_data(rg, in_f * out_f * k_y * k_x);

    std::vector<float> input_f32(input_bf16.size());
    std::vector<float> weights_f32(weights_bf16.size());
    for (size_t i = 0; i < input_bf16.size(); ++i) input_f32[i] = static_cast<float>(input_bf16[i]);
    for (size_t i = 0; i < weights_bf16.size(); ++i) weights_f32[i] = static_cast<float>(weights_bf16[i]);

    // BF16 path
    auto input_mem_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { batch, in_f, in_x, in_y } });
    // Deconv weights: { out_f, in_f, k_x, k_y } — note: oneDNN expects { in_f, out_f, kx, ky } for deconv
    auto weights_mem_bf16 = engine.allocate_memory({ data_types::bf16, format::bfyx, { out_f, in_f, k_x, k_y } });
    set_values(input_mem_bf16, input_bf16);
    set_values(weights_mem_bf16, weights_bf16);

    topology tp_bf16;
    tp_bf16.add(input_layout("input", input_mem_bf16->get_layout()));
    tp_bf16.add(data("weights", weights_mem_bf16));
    tp_bf16.add(deconvolution("deconv", input_info("input"), "weights", ""));
    tp_bf16.add(reorder("output", input_info("deconv"), format::bfyx, data_types::f32));

    auto result_bf16 = run_network_get_f32_output(engine, tp_bf16, "input", input_mem_bf16, "output", "deconv");

    // F32 reference
    auto input_mem_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, in_f, in_x, in_y } });
    auto weights_mem_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { out_f, in_f, k_x, k_y } });
    set_values(input_mem_f32, input_f32);
    set_values(weights_mem_f32, weights_f32);

    topology tp_f32;
    tp_f32.add(input_layout("input", input_mem_f32->get_layout()));
    tp_f32.add(data("weights", weights_mem_f32));
    tp_f32.add(deconvolution("deconv", input_info("input"), "weights", ""));
    tp_f32.add(reorder("output", input_info("deconv"), format::bfyx, data_types::f32));

    auto result_f32 = run_network_get_f32_output(engine, tp_f32, "input", input_mem_f32, "output");

    compare_outputs(result_f32, result_bf16, 0.05f);
}

#endif  // ENABLE_ONEDNN_FOR_GPU
