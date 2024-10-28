// Copyright (C) 2018-2024 Intel Corporation
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

using namespace cldnn;
using namespace ::tests;
using QuantizationType = ov::op::internal::DynamicQuantize::QuantizationType;

class dynamic_quantization_gpu_tests: public ::testing::Test {
public:

    void test_dynamic_quantization(bool is_caching_test,
                                   const ov::PartialShape& input_shape,
                                   const ov::Shape& data_shape,
                                   const QuantizationType quantization_type = QuantizationType::Symmetric,
                                   const std::string& impl_name = "") {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        auto input_ps = data_shape;
        auto dyn_input_ps = input_shape;
        auto scales_ps = ov::PartialShape::dynamic(dyn_input_ps.size());
        auto input_mem = engine.allocate_memory({ input_ps, data_types::f32, format::bfyx });
        auto group_sizes = std::vector<uint64_t>(dyn_input_ps.size(), 1);
        group_sizes.back() = UINT64_MAX;

        auto input_data = rg.generate_random_1d<float>(ov::shape_size(data_shape), -16.0f, 16.0f);
        set_values(input_mem, input_data);

        auto in_layout_f32 = input_shape.is_dynamic() ? layout{ dyn_input_ps, data_types::f32, format::bfyx }
                                                      : layout{ input_ps, data_types::f32, format::bfyx };

        auto in_layout = input_shape.is_dynamic() ? layout{ dyn_input_ps, data_types::f16, format::bfyx }
                                                  : layout{ input_ps, data_types::f16, format::bfyx };

        dynamic_quantize::Attributes dq_config;
        dq_config.quantization_type = quantization_type;
        dq_config.quantization_dt = data_types::i8;
        dq_config.scale_dt = data_types::f16;
        dq_config.zp_dt = data_types::undefined;
        dq_config.group_sizes = group_sizes;
        dq_config.scales_zp_output_order = { 0, 1, 2, 3 };
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        if (quantization_type == QuantizationType::Asymmetric) {
            dq_config.zp_dt = data_types::f16;
            dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;
        }

        auto reorder_1 = reorder("reorder_1", input_info("input"), layout{ input_ps, data_types::f16, format::bfyx });
        auto dyn_quan_prim = dynamic_quantize("dyn_quan_prim", input_info("reorder_1"), dq_config);
        auto reorder_data = reorder("reorder_data", input_info("dyn_quan_prim", 0), layout{ input_ps, data_types::f16, format::bfyx });
        auto reorder_scale = reorder("reorder_scale", input_info("dyn_quan_prim", 1), layout{ scales_ps, data_types::f16, format::bfyx });

        // Implemented dynamic quantize kernel
        auto get_ref_results = [&]() {
            topology topology(
                input_layout("input", in_layout_f32),
                reorder_1,
                dyn_quan_prim,
                reorder_data,
                reorder_scale
            );

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            config.set_property(ov::intel_gpu::optimize_data(true));

            ov::intel_gpu::ImplementationDesc dyn_quan_impl_desc = { format::bfyx, "dynamic_quantize_gpu_ref", impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"dyn_quan_prim", dyn_quan_impl_desc} }));

            network network(engine, topology, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();

            auto output_layout = outputs.begin()->second.get_layout();
            auto output_mem = outputs.begin()->second.get_memory();

            return engine.reinterpret_buffer(*output_mem, output_layout);
        };

        topology topology(
            input_layout("input", in_layout_f32),
            reorder_1,
            dyn_quan_prim,
            reorder_data
        );

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

        auto output_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<uint8_t> output_ptr (output_mem, get_test_stream());

        auto ref_output_mem = get_ref_results();
        cldnn::mem_lock<uint8_t> output_ptr_ref (ref_output_mem, get_test_stream());

        size_t count = 0;
        float max_diff = 0.f;
        float avg = 0.f;
        for (size_t i = 0; i < output_ptr_ref.size(); ++i) {
            auto abs_diff = std::abs(output_ptr_ref[i] - output_ptr[i]);
            if (max_diff < abs_diff)
                max_diff = abs_diff;
            avg += abs_diff;
            count++;
            OPENVINO_ASSERT(abs_diff < 1);
        }
        GPU_DEBUG_LOG << "---> count: " << count << ", max_diff:" << max_diff << ", avg_diff: " << (avg/count) << std::endl;
    }
};

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
    this->test_dynamic_quantization(false, {-1, 8, -1, 96}, {1, 8, 1, 96}, QuantizationType::Symmetric, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched) {
    this->test_dynamic_quantization(false, {-1, 4, -1, 64}, {1, 4, 35, 64}, QuantizationType::Symmetric, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_reordered) {
    this->test_dynamic_quantization(false, {-1, -1, 8, 96}, {1, 1, 8, 96}, QuantizationType::Symmetric, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched_reordered) {
    this->test_dynamic_quantization(false, {-1, -1, 4, 64}, {1, 35, 4, 64}, QuantizationType::Symmetric, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_asym) {
    this->test_dynamic_quantization(false, {-1, 8, -1, 96}, {1, 8, 1, 96}, QuantizationType::Asymmetric, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched_asym) {
    this->test_dynamic_quantization(false, {-1, 4, -1, 64}, {1, 4, 35, 64}, QuantizationType::Asymmetric, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_reordered_asym) {
    this->test_dynamic_quantization(false, {-1, -1, 8, 96}, {1, 1, 8, 96}, QuantizationType::Asymmetric, "dynamic_quantize_gpu_kv_cache");
}

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_kv_cache_batched_reordered_asym) {
    this->test_dynamic_quantization(false, {-1, -1, 4, 64}, {1, 35, 4, 64}, QuantizationType::Asymmetric, "dynamic_quantize_gpu_kv_cache");
}
