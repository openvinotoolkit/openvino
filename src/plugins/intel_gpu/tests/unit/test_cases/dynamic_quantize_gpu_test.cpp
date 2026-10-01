// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/tile.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/partial_shape.hpp"
#include "test_utils.h"
#include "random_generator.hpp"
#include "network_test.h"
#include <intel_gpu/runtime/utils.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include "intel_gpu/primitives/dynamic_quantize.hpp"
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "intel_gpu/runtime/compilation_context.hpp"
#include "fully_connected_inst.h"

#include <cmath>
#include <limits>

using namespace cldnn;
using namespace ::tests;
using QuantizationType = ov::op::internal::DynamicQuantize::QuantizationType;
using OutputStorageType = ov::op::internal::DynamicQuantize::OutputStorageType;

enum class SetInnerMostDimValuesZero { No, Yes };
enum class TestForSmallInputs { No, Yes };  // Test very small inputs to validate behavior in such small range
enum class PrecomputeSum { Disabled, Enabled };
class dynamic_quantization_gpu_tests: public ::testing::Test {
public:
    std::string dyn_quan_kernel_id = "";
    void test_dynamic_quantization(bool is_caching_test,
                                   const ov::PartialShape& input_shape,
                                   const ov::Shape& data_shape,
                                   const QuantizationType quantization_type = QuantizationType::Symmetric,
                                   uint64_t group_size = UINT64_MAX,
                                   data_types quant_dt = data_types::i8,
                                   data_types scale_dt = data_types::f16,
                                   data_types zp_dt = data_types::dynamic,
                                   OutputStorageType storage_type = OutputStorageType::Planar,
                                   const std::string& impl_name = "",
                                   SetInnerMostDimValuesZero set_inner_most_dim_values_zero = SetInnerMostDimValuesZero::No,
                                   const PrecomputeSum has_precompute_sum = PrecomputeSum::Disabled,
                                   const TestForSmallInputs test_for_small_inputs = TestForSmallInputs::No) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        auto input_ps = data_shape;
        auto dyn_input_ps = input_shape;
        auto scales_ps = ov::PartialShape::dynamic(dyn_input_ps.size());
        auto input_mem = engine.allocate_memory({ input_ps, data_types::f32, format::bfyx });
        auto group_sizes = std::vector<uint64_t>(dyn_input_ps.size(), 1);
        group_sizes.back() = group_size;

        auto input_data = rg.generate_random_1d<float>(ov::shape_size(data_shape), -16.0f, 20.0f);

        // Test for very small values to check the behavior where input values are like 0.001.
        // In such case, finding max and converting it to int may behave differently due to fp16 range and calculation structure in cl file
        if (test_for_small_inputs == TestForSmallInputs::Yes)
            std::transform(input_data.begin(), input_data.end(), input_data.begin(),
                       [](float val) { return val / 10000.0f; });

        if (set_inner_most_dim_values_zero == SetInnerMostDimValuesZero::Yes)
           std::fill(input_data.begin(), input_data.begin() + data_shape[data_shape.size() - 1], 0.0f);
        set_values(input_mem, input_data);

        auto in_layout_f32 = input_shape.is_dynamic() ? layout{ dyn_input_ps, data_types::f32, format::bfyx }
                                                      : layout{ input_ps, data_types::f32, format::bfyx };

        auto in_layout = input_shape.is_dynamic() ? layout{ dyn_input_ps, data_types::f16, format::bfyx }
                                                  : layout{ input_ps, data_types::f16, format::bfyx };

        dynamic_quantize::Attributes dq_config;
        dq_config.quantization_type = quantization_type;
        dq_config.quantization_dt = quant_dt;
        dq_config.scale_dt = scale_dt;
        dq_config.zp_dt = zp_dt;
        dq_config.group_sizes = group_sizes;
        dq_config.scales_zp_output_order = { 0, 1, 2};

        if (has_precompute_sum == PrecomputeSum::Enabled) {
            dq_config.precomputed_reduction = true;
            dq_config.precomputed_reduction_dt = data_types::i32;
        }

        if (data_shape.size() == 4)
            dq_config.scales_zp_output_order.emplace_back(3);
        dq_config.output_storage_type = storage_type;

        bool has_zp_output = dq_config.quantization_type == QuantizationType::Asymmetric &&
                             dq_config.output_storage_type == OutputStorageType::Planar;

        auto reorder_1 = reorder("reorder_1", input_info("input"), layout{ input_ps, data_types::f16, format::bfyx });
        auto dyn_quan_prim = dynamic_quantize("dyn_quan_prim", input_info("reorder_1"), dq_config);
        auto reorder_data = reorder("reorder_data", input_info("dyn_quan_prim", 0), layout{ input_ps, data_types::f16, format::bfyx });
        auto reorder_scale = reorder("reorder_scale", input_info("dyn_quan_prim", 1), layout{ scales_ps, data_types::f16, format::bfyx });
        // The kernel does not support zp and precompute_sum at the same time.
        auto reorder_zp = reorder("reorder_zp", input_info("dyn_quan_prim", 2), layout{ scales_ps, data_types::f16, format::bfyx });
        auto reorder_precompute_sum = reorder("reorder_precompute_sum", input_info("dyn_quan_prim", 2), layout{ scales_ps, data_types::f16, format::bfyx });

        // Implemented dynamic quantize kernel
        auto get_ref_results = [&]() {
            topology topology(
                input_layout("input", in_layout_f32),
                reorder_1,
                dyn_quan_prim,
                reorder_data,
                reorder_scale
            );

            if (has_zp_output)
                topology.add(reorder_zp);

            if (has_precompute_sum == PrecomputeSum::Enabled)
                topology.add(reorder_precompute_sum);

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            config.set_property(ov::intel_gpu::optimize_data(true));

            ov::intel_gpu::ImplementationDesc dyn_quan_impl_desc = { format::bfyx, "dynamic_quantize_gpu_ref", impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"dyn_quan_prim", dyn_quan_impl_desc} }));

            network network(engine, topology, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();

            std::vector<memory::ptr> output_buffers;
            for (const auto& output : outputs) {
                auto output_layout = output.second.get_layout();
                auto output_mem = output.second.get_memory();
                output_buffers.push_back(engine.reinterpret_buffer(*output_mem, output_layout));
            }

            return output_buffers;
        };

        topology topology(
            input_layout("input", in_layout_f32),
            reorder_1,
            dyn_quan_prim,
            reorder_data,
            reorder_scale
        );

        if (has_zp_output)
            topology.add(reorder_zp);

        if (has_precompute_sum == PrecomputeSum::Enabled) {
            ASSERT_EQ(has_zp_output, false);    // the kernel does not support zp and precompute_sum at the same time
            topology.add(reorder_precompute_sum);
        }

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));

        if (impl_name != "") {
            ov::intel_gpu::ImplementationDesc dyn_quan_impl_desc = { format::bfyx, impl_name, impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"dyn_quan_prim", dyn_quan_impl_desc} }));
        }

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();

        std::vector<memory::ptr> output_buffers;
        for (const auto& output : outputs) {
            auto output_layout = output.second.get_layout();
            auto output_mem = output.second.get_memory();
            output_buffers.push_back(engine.reinterpret_buffer(*output_mem, output_layout));
        }

        auto ref_output_buffers = get_ref_results();

        ASSERT_EQ(ref_output_buffers.size(), output_buffers.size());


        for (size_t i = 0; i < ref_output_buffers.size(); i++) {
            cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_buffers[i], get_test_stream());
            cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr_ref(ref_output_buffers[i], get_test_stream());

            for (size_t j = 0; j < output_ptr_ref.size(); ++j) {
                if (ov::element::Type(quant_dt).is_real()) {
                    auto abs_diff = std::abs(output_ptr_ref[j] - output_ptr[j]);
                    ASSERT_EQ(abs_diff, 0);
                } else { // (u)int8
                    const int abs_error_threshold = 2;
                    ASSERT_NEAR(output_ptr_ref[j], output_ptr[j], abs_error_threshold);
                }
            }
        }

        auto find_kernel_id = [&network](std::string prim_id) {
                std::string kernel = "";
                for (auto& info : network->get_primitives_info()) {
                        if (info.original_id == prim_id)
                            kernel = info.kernel_id;
                }
                return kernel;
            };

        dyn_quan_kernel_id = find_kernel_id("dyn_quan_prim");
    }
};

TEST_F(dynamic_quantization_gpu_tests, static_quantizing_large_size_non_uniform_workgroup) {
    // dispatch_block_num is always aligned to block_num (dynamic_quantize_kernel_opt.cpp),
    // so the opt kernel's dispatch is safe regardless of non-uniform work-group support.
    this->test_dynamic_quantization(false, {11, 1, 4096+128}, {2048, 1, 4096+128}, QuantizationType::Symmetric, 128);
    ASSERT_TRUE(dyn_quan_kernel_id.find("_opt") != std::string::npos);
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_large_size) {
    this->test_dynamic_quantization(false, {11, 1, 1, 4096}, {2048, 1, 1, 4096});
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_large_size_dynamic) {
    this->test_dynamic_quantization(false, {-1, 1, 1, 4096}, {2048, 1, 1, 4096});
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_small_size) {
    this->test_dynamic_quantization(false, {1, 1, 1, 4096}, {64, 1, 1, 4096});
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_single_batch) {
    this->test_dynamic_quantization(false, {-1, 1, 1, 4096}, {1, 1, 1, 4096});
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_small_size_precompute_gs32) {
    this->test_dynamic_quantization(false, {1, 1, 512},
                                    {32, 1, 512},
                                    QuantizationType::Symmetric,
                                    32,
                                    data_types::i8,
                                    data_types::f16,
                                    data_types::i8,
                                    OutputStorageType::Planar,
                                    "",
                                    SetInnerMostDimValuesZero::No,
                                    PrecomputeSum::Enabled);
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_small_size_precompute_gs128) {
    this->test_dynamic_quantization(false, {1, 1, 512},
                                    {32, 1, 512},
                                    QuantizationType::Symmetric,
                                    128,
                                    data_types::i8,
                                    data_types::f16,
                                    data_types::i8,
                                    OutputStorageType::Planar,
                                    "",
                                    SetInnerMostDimValuesZero::No,
                                    PrecomputeSum::Enabled);
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_small_size_precompute_gs128_cache) {
    this->test_dynamic_quantization(true, {-1, 1, 512},
                                    {32, 1, 512},
                                    QuantizationType::Symmetric,
                                    128,
                                    data_types::i8,
                                    data_types::f16,
                                    data_types::i8,
                                    OutputStorageType::Planar,
                                    "",
                                    SetInnerMostDimValuesZero::No,
                                    PrecomputeSum::Enabled);
}


TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_small_size_precompute_gs128_small_values) {
    this->test_dynamic_quantization(false, {1, 1, 512},
                                    {32, 1, 512},
                                    QuantizationType::Symmetric,
                                    128,
                                    data_types::i8,
                                    data_types::f16,
                                    data_types::i8,
                                    OutputStorageType::Planar,
                                    "",
                                    SetInnerMostDimValuesZero::No,
                                    PrecomputeSum::Enabled,
                                    TestForSmallInputs::Yes);
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_asym_act) {
    this->test_dynamic_quantization(false, {-1, 1, 1, 4096}, {1, 1, 1, 4096}, QuantizationType::Asymmetric, UINT64_MAX,
                                    data_types::u8, data_types::f16, data_types::u8, OutputStorageType::Planar);
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_small_size_grouped) {
    this->test_dynamic_quantization(false, {1, 1, 4096}, {64, 1, 4096}, QuantizationType::Symmetric, 32);
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_small_size_gs128) {
    this->test_dynamic_quantization(false, {1, 1, 4096}, {64, 1, 4096}, QuantizationType::Symmetric, 128);
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_single_batch_grouped) {
    this->test_dynamic_quantization(false, {-1, 1, 4096}, {1, 1, 4096}, QuantizationType::Symmetric, 32);
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_ref_only) {
    this->test_dynamic_quantization(false, {-1, 1, 1, 33}, {16, 1, 1, 33});
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_ref_only_dynamic) {
    this->test_dynamic_quantization(false, {1, 1, 1, 33}, {16, 1, 1, 33});
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_invalid) {
    this->test_dynamic_quantization(false, {-1, 1, 1, 7}, {16, 1, 1, 7});
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_unaligned) {
    this->test_dynamic_quantization(false, {-1, 1, 1, 32}, {16, 1, 1, 32});
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_unaligned_dynamic) {
    this->test_dynamic_quantization(false, {1, 1, 1, 32}, {16, 1, 1, 32});
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache) {
    this->test_dynamic_quantization(false,
                                    {-1, 8, -1, 96},
                                    {1, 8, 1, 96},
                                    QuantizationType::Symmetric,
                                    UINT64_MAX,
                                    data_types::i8,
                                    data_types::f16,
                                    data_types::dynamic,
                                    OutputStorageType::Planar,
                                    "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched) {
    this->test_dynamic_quantization(false,
                                    {-1, 4, -1, 64},
                                    {1, 4, 35, 64},
                                    QuantizationType::Symmetric,
                                    UINT64_MAX,
                                    data_types::i8,
                                    data_types::f16,
                                    data_types::dynamic,
                                    OutputStorageType::Planar,
                                    "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_reordered) {
    this->test_dynamic_quantization(false,
                                    {-1, -1, 8, 96},
                                    {1, 1, 8, 96},
                                    QuantizationType::Symmetric,
                                    UINT64_MAX,
                                    data_types::i8,
                                    data_types::f16,
                                    data_types::dynamic,
                                    OutputStorageType::Planar,
                                    "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched_reordered) {
    this->test_dynamic_quantization(false,
                                    {-1, -1, 4, 64},
                                    {1, 35, 4, 64},
                                    QuantizationType::Symmetric,
                                    UINT64_MAX,
                                    data_types::i8,
                                    data_types::f16,
                                    data_types::dynamic,
                                    OutputStorageType::Planar,
                                    "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_asym_planar) {
    this->test_dynamic_quantization(false, {-1, 8, -1, 96}, {1, 8, 1, 96}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::f16, OutputStorageType::Planar, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched_asym_planar) {
    this->test_dynamic_quantization(false, {-1, 4, -1, 64}, {1, 4, 35, 64}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::f16, OutputStorageType::Planar, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_reordered_asym_planar) {
    this->test_dynamic_quantization(false, {-1, -1, 8, 96}, {1, 1, 8, 96}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::f16, OutputStorageType::Planar, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched_reordered_asym_planar) {
    this->test_dynamic_quantization(false, {-1, -1, 4, 64}, {1, 35, 4, 64}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::f16, OutputStorageType::Planar, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_asym_interleaved) {
    this->test_dynamic_quantization(false, {-1, 8, -1, 96}, {1, 8, 1, 96}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::f16, OutputStorageType::InterleavedScalesZP, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched_asym_interleaved) {
    this->test_dynamic_quantization(false, {-1, 4, -1, 64}, {1, 4, 35, 64}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::f16, OutputStorageType::InterleavedScalesZP, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_reordered_asym_interleaved) {
    this->test_dynamic_quantization(false, {-1, -1, 8, 96}, {1, 1, 8, 96}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::f16, OutputStorageType::InterleavedScalesZP, "dynamic_quantize_gpu_kv_cache");
}


TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched_reordered_asym_interleaved) {
    this->test_dynamic_quantization(false, {-1, -1, 4, 64}, {1, 35, 4, 64}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::f16, OutputStorageType::InterleavedScalesZP, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_asym_planar_i8_zp) {
    this->test_dynamic_quantization(false, {-1, 8, -1, 32}, {1, 8, 1, 32}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::i8, OutputStorageType::Planar, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_asym_planar_i8_zp_head512) {
    this->test_dynamic_quantization(false, {-1, 4, -1, 512}, {1, 4, 35, 512}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::i8, OutputStorageType::Planar, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched_asym_planar_i8_zp) {
    this->test_dynamic_quantization(false, {-1, 4, -1, 64}, {1, 4, 35, 64}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::i8, OutputStorageType::Planar, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_reordered_asym_planar_i8_zp) {
    this->test_dynamic_quantization(false, {-1, -1, 8, 96}, {1, 1, 8, 96}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::i8, OutputStorageType::Planar, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched_reordered_asym_planar_i8_zp) {
    this->test_dynamic_quantization(false, {-1, -1, 4, 64}, {1, 35, 4, 64}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::i8, OutputStorageType::Planar, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_inner_most_dim_zero_values_asym) {
    this->test_dynamic_quantization(false, {-1, 8, -1, 128}, {1, 8, 52, 128}, QuantizationType::Asymmetric, UINT64_MAX,
                                data_types::i8, data_types::f16, data_types::f16, OutputStorageType::InterleavedScalesZP, "dynamic_quantize_gpu_kv_cache", SetInnerMostDimValuesZero::Yes);
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantization_mxf8e4m3) {
    this->test_dynamic_quantization(false,
                                    {1, 1, 4096},
                                    {1, 1, 4096},
                                    QuantizationType::Symmetric,
                                    32,
                                    data_types::f8e4m3,
                                    data_types::f8e8m0,
                                    data_types::dynamic,
                                    OutputStorageType::Planar);
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantization_mxf8e5m2) {
    this->test_dynamic_quantization(false,
                                    {1, 1, 4096},
                                    {1, 1, 4096},
                                    QuantizationType::Symmetric,
                                    32,
                                    data_types::f8e5m2,
                                    data_types::f8e8m0,
                                    data_types::dynamic,
                                    OutputStorageType::Planar);
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantization_f8e4m3) {
    this->test_dynamic_quantization(false,
                                    {1, 1, 4096},
                                    {1, 1, 4096},
                                    QuantizationType::Symmetric,
                                    UINT64_MAX,
                                    data_types::f8e4m3,
                                    data_types::f16,
                                    data_types::dynamic,
                                    OutputStorageType::Planar);
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantization_f8e5m2) {
    this->test_dynamic_quantization(false,
                                    {1, 1, 4096},
                                    {1, 1, 4096},
                                    QuantizationType::Symmetric,
                                    UINT64_MAX,
                                    data_types::f8e5m2,
                                    data_types::f16,
                                    data_types::dynamic,
                                    OutputStorageType::Planar);
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantization_mxf4e2m1) {
    this->test_dynamic_quantization(false,
                                    {1, 128, 1, 32},
                                    {1, 128, 1, 32},
                                    QuantizationType::Symmetric,
                                    32,
                                    data_types::f4e2m1,
                                    data_types::f8e8m0,
                                    data_types::dynamic,
                                    OutputStorageType::Planar);
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantization_f4e2m1) {
    this->test_dynamic_quantization(false,
                                    {1, 1, 4096},
                                    {1, 1, 4096},
                                    QuantizationType::Symmetric,
                                    UINT64_MAX,
                                    data_types::f4e2m1,
                                    data_types::f16,
                                    data_types::dynamic,
                                    OutputStorageType::Planar);
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantize_opt_group_size_256) {
    this->test_dynamic_quantization(false, {1, 1, 8192}, {1, 1, 8192}, QuantizationType::Symmetric, 256,
                                data_types::i8, data_types::f16, data_types::dynamic, OutputStorageType::Planar,
                                "dynamic_quantize_gpu_opt");
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantize_opt_group_size_256_precompute_sum) {
    this->test_dynamic_quantization(false, {1, 1, 8192}, {1, 1, 8192}, QuantizationType::Symmetric, 256,
                                data_types::i8, data_types::f16, data_types::i8, OutputStorageType::Planar,
                                "dynamic_quantize_gpu_opt", SetInnerMostDimValuesZero::No, PrecomputeSum::Enabled);
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantize_group_size_8192_with_precompute_sum) {
    this->test_dynamic_quantization(false, {1, 1, 16384}, {1, 1, 16384}, QuantizationType::Symmetric, 8192,
                                data_types::i8, data_types::f16, data_types::i8, OutputStorageType::Planar,
                                "", SetInnerMostDimValuesZero::No, PrecomputeSum::Enabled);
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantize_opt_gs128_K2560_sym_precompute_sum) {
    this->test_dynamic_quantization(false, {1, 1, 2560}, {1, 1, 2560}, QuantizationType::Symmetric, 128,
                                data_types::i8, data_types::f16, data_types::i8, OutputStorageType::Planar,
                                "dynamic_quantize_gpu_opt", SetInnerMostDimValuesZero::No,
                                PrecomputeSum::Enabled);
}

// GS=64: LARGE_GS mode, GS=32/16: SMALL_GS mode
TEST_F(dynamic_quantization_gpu_tests, dynamic_quantize_opt_gs64) {
    this->test_dynamic_quantization(false, {-1, 1, 4096}, {1, 1, 4096}, QuantizationType::Symmetric, 64,
                                data_types::i8, data_types::f16, data_types::dynamic, OutputStorageType::Planar,
                                "dynamic_quantize_gpu_opt");
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantize_opt_gs32) {
    this->test_dynamic_quantization(false, {-1, 1, 4096}, {1, 1, 4096}, QuantizationType::Symmetric, 32,
                                data_types::i8, data_types::f16, data_types::dynamic, OutputStorageType::Planar,
                                "dynamic_quantize_gpu_opt");
}

TEST_F(dynamic_quantization_gpu_tests, dynamic_quantize_opt_gs16) {
    this->test_dynamic_quantization(false, {-1, 1, 4096}, {1, 1, 4096}, QuantizationType::Symmetric, 16,
                                data_types::i8, data_types::f16, data_types::dynamic, OutputStorageType::Planar,
                                "dynamic_quantize_gpu_opt");
}

// For bf16 the input storage type is a raw ushort, so the kernels have to decode bf16 (instead of
// relying on an implicit type conversion, which would reinterpret the bit pattern, e.g. 0.5 -> 16128).
// The scale is derived from the decoded group min/max, so it is compared against a value computed on
// the host from the very same bf16 data (comparing GPU kernels against each other would not catch a
// decoding bug shared by all of them).
// With beyond_f16_range some elements exceed the f16 range (only representable for bf16 input), so the
// kernels must not narrow the decoded bf16 values to half (|x| > 65504 would turn into inf and give inf/NaN
// scales). The resulting scales (max(|x|) / 127) still fit into the f16 scale storage.
static void test_scale_is_computed_from_decoded_values(const std::string& impl_name,
                                                       ov::element::Type input_type,
                                                       uint64_t group_size,
                                                       bool beyond_f16_range) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    const ov::Shape data_shape = {1, 1, 4096};
    const bool per_token = group_size == std::numeric_limits<uint64_t>::max();
    const size_t elements_per_group = per_token ? data_shape.back() : static_cast<size_t>(group_size);
    const size_t groups_num = ov::shape_size(data_shape) / elements_per_group;

    // Round trip the data through the input precision first so that the host reference below
    // sees exactly the values the kernel reads
    auto round_to_input_type = [&](float value) {
        return input_type == ov::element::bf16 ? static_cast<float>(ov::bfloat16(value))
                                               : static_cast<float>(ov::float16(value));
    };
    std::vector<float> host_values;
    for (auto value : rg.generate_random_1d<float>(ov::shape_size(data_shape), -16.0f, 20.0f)) {
        host_values.push_back(round_to_input_type(value));
    }
    // Make sure the groups have different maxima so that misaligned scales are detectable
    for (size_t g = 0; g < groups_num; g++) {
        host_values[g * elements_per_group] = 1.0f + static_cast<float>(g);
    }
    if (beyond_f16_range) {
        // Put a few values beyond the f16 range (max 65504) into every other group, with both signs
        for (size_t g = 0; g < groups_num; g += 2) {
            const float magnitude = 1.0e5f + 2.0e5f * static_cast<float>(g) / static_cast<float>(groups_num);
            host_values[g * elements_per_group + 1] = round_to_input_type(magnitude);
            host_values[g * elements_per_group + elements_per_group - 1] = round_to_input_type(-1.5f * magnitude);
        }
    }

    const data_types input_dt = input_type == ov::element::bf16 ? data_types::bf16 : data_types::f16;

    auto input_mem = engine.allocate_memory({data_shape, input_dt, format::bfyx});
    if (input_type == ov::element::bf16) {
        std::vector<ov::bfloat16> data;
        for (auto value : host_values) {
            data.emplace_back(value);
        }
        set_values(input_mem, data);
    } else {
        std::vector<ov::float16> data;
        for (auto value : host_values) {
            data.emplace_back(value);
        }
        set_values(input_mem, data);
    }

    // Symmetric i8 per-group quantization: one scale per group holding max(|x|) / 127
    std::vector<float> expected_scales(groups_num, 0.0f);
    for (size_t g = 0; g < groups_num; g++) {
        float group_max = 0.0f;
        for (size_t e = 0; e < elements_per_group; e++) {
            group_max = std::max(group_max, std::abs(host_values[g * elements_per_group + e]));
        }
        expected_scales[g] = group_max / 127.0f;
    }

    dynamic_quantize::Attributes dq_config;
    dq_config.quantization_type = QuantizationType::Symmetric;
    dq_config.quantization_dt = data_types::i8;
    dq_config.scale_dt = data_types::f16;
    dq_config.zp_dt = data_types::dynamic;
    dq_config.group_sizes = {1, 1, group_size};
    dq_config.scales_zp_output_order = {0, 1, 2};
    dq_config.output_storage_type = OutputStorageType::Planar;

    topology topology(
        input_layout("input", layout{data_shape, input_dt, format::bfyx}),
        dynamic_quantize("dyn_quan_prim", input_info("input"), dq_config),
        reorder("out_data", input_info("dyn_quan_prim", 0), layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}),
        reorder("out_scale", input_info("dyn_quan_prim", 1), layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"dyn_quan_prim", { format::bfyx, impl_name, impl_types::ocl }}}));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    ASSERT_TRUE(outputs.count("out_scale") > 0);
    ASSERT_TRUE(outputs.count("out_data") > 0);
    auto scale_mem = outputs.at("out_scale").get_memory();
    auto data_mem = outputs.at("out_data").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> scale_ptr(scale_mem, get_test_stream());
    cldnn::mem_lock<float, mem_lock_type::read> data_ptr(data_mem, get_test_stream());

    const auto scale_layout = outputs.at("out_scale").get_layout();
    ASSERT_GE(scale_ptr.size(), groups_num)
        << "impl: " << impl_name << ", scale layout: " << scale_layout.to_string();
    ASSERT_EQ(data_ptr.size(), host_values.size());
    for (size_t g = 0; g < groups_num; g++) {
        const float actual_scale = static_cast<float>(scale_ptr[g]);
        ASSERT_TRUE(std::isfinite(actual_scale))
            << "impl: " << impl_name << ", group: " << g << ", actual: " << actual_scale;
        ASSERT_NEAR(actual_scale, expected_scales[g], expected_scales[g] * 0.05f)
            << "impl: " << impl_name << ", group: " << g << ", actual: " << actual_scale
            << ", expected: " << expected_scales[g];
        // The quantized values have to match the host values scaled by the expected scale
        for (size_t e = 0; e < elements_per_group; e++) {
            const size_t idx = g * elements_per_group + e;
            ASSERT_NEAR(data_ptr[idx], host_values[idx] / expected_scales[g], 1.0f)
                << "impl: " << impl_name << ", group: " << g << ", element: " << e;
        }
    }
}

class dynamic_quantization_bf16_input_tests : public dynamic_quantization_gpu_tests,
                                              public ::testing::WithParamInterface<std::tuple<std::string, ov::element::Type>> {};

TEST_P(dynamic_quantization_bf16_input_tests, scale_is_computed_from_decoded_values) {
    test_scale_is_computed_from_decoded_values(std::get<0>(GetParam()), std::get<1>(GetParam()), 64, false);
}

INSTANTIATE_TEST_SUITE_P(bf16_input,
                         dynamic_quantization_bf16_input_tests,
                         ::testing::Combine(::testing::Values("dynamic_quantize_gpu_ref", "dynamic_quantize_gpu_opt"),
                                            ::testing::Values(ov::element::f16, ov::element::bf16)),
                         [](const ::testing::TestParamInfo<std::tuple<std::string, ov::element::Type>>& info) {
                             return (std::get<0>(info.param).find("_ref") != std::string::npos ? "ref_" : "opt_") +
                                    std::get<1>(info.param).get_type_name();
                         });

// bf16 only: values beyond the f16 range. Group sizes cover the small group, large group and per token
// paths of the opt kernel.
class dynamic_quantization_bf16_beyond_f16_range_tests : public dynamic_quantization_gpu_tests,
                                                         public ::testing::WithParamInterface<std::tuple<std::string, uint64_t>> {};

TEST_P(dynamic_quantization_bf16_beyond_f16_range_tests, scale_is_finite) {
    test_scale_is_computed_from_decoded_values(std::get<0>(GetParam()), ov::element::bf16, std::get<1>(GetParam()), true);
}

INSTANTIATE_TEST_SUITE_P(bf16_input,
                         dynamic_quantization_bf16_beyond_f16_range_tests,
                         ::testing::Combine(::testing::Values("dynamic_quantize_gpu_ref", "dynamic_quantize_gpu_opt"),
                                            ::testing::Values(uint64_t{32}, uint64_t{64}, std::numeric_limits<uint64_t>::max())),
                         [](const ::testing::TestParamInfo<std::tuple<std::string, uint64_t>>& info) {
                             const auto group_size = std::get<1>(info.param);
                             return std::string(std::get<0>(info.param).find("_ref") != std::string::npos ? "ref_" : "opt_") +
                                    (group_size == std::numeric_limits<uint64_t>::max() ? std::string("per_token")
                                                                                         : "gs" + std::to_string(group_size));
                         });

// bf16 input with bf16 scales (and fp zero points), as used for the KV cache of bf16 models. bf16 scale/zp
// buffers are stored as ushort, so the kernels have to encode them as bf16 (not with a numeric cast).
// Groups alternate between large magnitudes (up to 1e7, f16 scales would overflow or lose all precision)
// and small ones; scales, zero points and dequantized values are checked against the host data.
struct dq_bf16_scales_params {
    std::string impl_name;
    QuantizationType quantization_type;
    OutputStorageType storage_type;
    data_types zp_dt;
};

class dynamic_quantization_bf16_scales_tests : public dynamic_quantization_gpu_tests,
                                               public ::testing::WithParamInterface<dq_bf16_scales_params> {};

TEST_P(dynamic_quantization_bf16_scales_tests, scales_and_zp_are_stored_as_bf16) {
    const auto& p = GetParam();
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    const bool is_asym = p.quantization_type == QuantizationType::Asymmetric;
    const bool is_interleaved = p.storage_type == OutputStorageType::InterleavedScalesZP;
    const bool has_zp_output = is_asym && !is_interleaved;

    const ov::Shape data_shape = {1, 4, 8, 64};
    const size_t elements_per_group = data_shape.back();
    const size_t groups_num = ov::shape_size(data_shape) / elements_per_group;
    const std::vector<float> magnitudes = {1.0e5f, 2.5e5f, 1.0e6f, 7.0e6f, 1.0e7f, 3.0f, 0.5f, 20.0f};

    // Asymmetric groups are shifted, so that the zero point is not trivial
    std::vector<float> host_values;
    for (size_t g = 0; g < groups_num; g++) {
        const float magnitude = magnitudes[g % magnitudes.size()];
        const float low = is_asym ? -0.5f * magnitude : -magnitude;
        for (auto value : rg.generate_random_1d<float>(elements_per_group, low, magnitude)) {
            host_values.push_back(static_cast<float>(ov::bfloat16(value)));
        }
    }

    auto input_mem = engine.allocate_memory({data_shape, data_types::bf16, format::bfyx});
    std::vector<ov::bfloat16> input_data;
    for (auto value : host_values) {
        input_data.emplace_back(value);
    }
    set_values(input_mem, input_data);

    dynamic_quantize::Attributes dq_config;
    dq_config.quantization_type = p.quantization_type;
    dq_config.quantization_dt = data_types::i8;
    dq_config.scale_dt = data_types::bf16;
    dq_config.zp_dt = is_asym ? p.zp_dt : data_types::dynamic;
    dq_config.group_sizes = {1, 1, 1, UINT64_MAX};
    dq_config.scales_zp_output_order = {0, 1, 2, 3};
    dq_config.output_storage_type = p.storage_type;

    const auto dyn_f32_layout = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    topology topology(input_layout("input", layout{{-1, 4, -1, 64}, data_types::bf16, format::bfyx}),
                      dynamic_quantize("dyn_quan_prim", input_info("input"), dq_config),
                      reorder("out_data", input_info("dyn_quan_prim", 0), dyn_f32_layout),
                      reorder("out_scale", input_info("dyn_quan_prim", 1), dyn_f32_layout));
    if (has_zp_output) {
        topology.add(reorder("out_zp", input_info("dyn_quan_prim", 2), dyn_f32_layout));
    }

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"dyn_quan_prim", {format::bfyx, p.impl_name, impl_types::ocl}}}));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    auto data_mem = outputs.at("out_data").get_memory();
    auto scale_mem = outputs.at("out_scale").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> data_ptr(data_mem, get_test_stream());
    cldnn::mem_lock<float, mem_lock_type::read> scale_ptr(scale_mem, get_test_stream());
    ASSERT_EQ(data_ptr.size(), host_values.size());
    ASSERT_EQ(scale_ptr.size(), groups_num * (is_interleaved ? 2 : 1));

    std::vector<float> zp_values(groups_num, 0.0f);
    if (has_zp_output) {
        auto zp_mem = outputs.at("out_zp").get_memory();
        cldnn::mem_lock<float, mem_lock_type::read> zp_ptr(zp_mem, get_test_stream());
        ASSERT_EQ(zp_ptr.size(), groups_num);
        for (size_t g = 0; g < groups_num; g++) {
            zp_values[g] = zp_ptr[g];
        }
    } else if (is_interleaved) {
        for (size_t g = 0; g < groups_num; g++) {
            zp_values[g] = scale_ptr[g * 2 + 1];
        }
    }

    for (size_t g = 0; g < groups_num; g++) {
        float group_min = std::numeric_limits<float>::max();
        float group_max = std::numeric_limits<float>::lowest();
        float group_abs_max = 0.0f;
        for (size_t e = 0; e < elements_per_group; e++) {
            const float value = host_values[g * elements_per_group + e];
            group_min = std::min(group_min, value);
            group_max = std::max(group_max, value);
            group_abs_max = std::max(group_abs_max, std::abs(value));
        }

        // Dequantization scale and zero point: x ~= (q - zp) * scale
        const float expected_scale = is_asym ? (group_max - group_min) / 255.0f : group_abs_max / 127.0f;
        const float expected_zp = is_asym ? -group_min / expected_scale - 128.0f : 0.0f;
        const float actual_scale = scale_ptr[is_interleaved ? g * 2 : g];
        const float actual_zp = zp_values[g];

        ASSERT_TRUE(std::isfinite(actual_scale)) << "group: " << g;
        // bf16 has an 8 bit mantissa (relative rounding error <= 2^-9)
        ASSERT_NEAR(actual_scale, expected_scale, expected_scale * 0.01f)
            << "group: " << g << ", actual: " << actual_scale << ", expected: " << expected_scale;
        // fp zero points are rounded to bf16 (step 0.5 for |zp| in [64, 128)), i8 zero points to an integer
        ASSERT_NEAR(actual_zp, expected_zp, 1.0f)
            << "group: " << g << ", actual: " << actual_zp << ", expected: " << expected_zp;

        for (size_t e = 0; e < elements_per_group; e++) {
            const size_t idx = g * elements_per_group + e;
            const float dequantized = (data_ptr[idx] - actual_zp) * actual_scale;
            ASSERT_NEAR(dequantized, host_values[idx], 2.0f * expected_scale)
                << "group: " << g << ", element: " << e << ", q: " << data_ptr[idx];
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    bf16_scales,
    dynamic_quantization_bf16_scales_tests,
    ::testing::Values(
        dq_bf16_scales_params{"dynamic_quantize_gpu_kv_cache", QuantizationType::Symmetric, OutputStorageType::Planar, data_types::dynamic},
        dq_bf16_scales_params{"dynamic_quantize_gpu_kv_cache", QuantizationType::Asymmetric, OutputStorageType::Planar, data_types::bf16},
        dq_bf16_scales_params{"dynamic_quantize_gpu_kv_cache", QuantizationType::Asymmetric, OutputStorageType::Planar, data_types::i8},
        dq_bf16_scales_params{"dynamic_quantize_gpu_kv_cache", QuantizationType::Asymmetric, OutputStorageType::InterleavedScalesZP, data_types::bf16},
        dq_bf16_scales_params{"dynamic_quantize_gpu_ref", QuantizationType::Symmetric, OutputStorageType::Planar, data_types::dynamic},
        dq_bf16_scales_params{"dynamic_quantize_gpu_ref", QuantizationType::Asymmetric, OutputStorageType::Planar, data_types::bf16},
        dq_bf16_scales_params{"dynamic_quantize_gpu_ref", QuantizationType::Asymmetric, OutputStorageType::Planar, data_types::i8},
        dq_bf16_scales_params{"dynamic_quantize_gpu_ref", QuantizationType::Asymmetric, OutputStorageType::InterleavedScalesZP, data_types::bf16}),
    [](const ::testing::TestParamInfo<dq_bf16_scales_params>& info) {
        const auto& p = info.param;
        std::string name = p.impl_name.find("_ref") != std::string::npos ? "ref" : "kv_cache";
        if (p.quantization_type == QuantizationType::Symmetric) {
            return name + "_sym";
        }
        name += p.storage_type == OutputStorageType::InterleavedScalesZP ? "_asym_interleaved" : "_asym_planar";
        return name + (p.zp_dt == data_types::i8 ? "_i8_zp" : "_bf16_zp");
    });
