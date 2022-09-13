// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/pooling.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct pooling_test_params {
    tensor in_shape;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    pooling_mode pool_mode;
    std::string kernel_name;
};

class PoolingFusingTest : public ::BaseFusingTest<pooling_test_params> {
public:
    void execute(pooling_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        build_options options;
        options.set_option(build_option::optimize_data(true));
        if (!p.kernel_name.empty()) {
            implementation_desc impl = { p.input_format, p.kernel_name };
            options.set_option(build_option::force_implementations({ { "pooling", impl } }));
        }
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, options);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        ASSERT_FALSE(network_fused.get_primitives_info().empty());
        ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

        auto find_and_check = [&](primitive_info& p) -> bool {
            if (p.original_id == "pooling" || p.original_id == "output_reorder")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto pi_not_fused = network_not_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_and_check);
        auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_and_check);

        ASSERT_TRUE(info_fused != pi_fused.end());
        ASSERT_TRUE(info_not_fused != pi_not_fused.end());

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(pooling_test_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape };
    }

    layout get_per_channel_layout(pooling_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.in_shape.feature[0], 1, 1 } };
    }
};

}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------------- Pooling cases ----------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_POOLING_F32_1 { 1, 16, 8, 8 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_F32_2 { 2, 16, 8, 8 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_F32_3 { 1, 32, 10, 10 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_F32_4 { 1, 32, 10, 10 }, data_types::f32, format::fs_b_yx_fsv32, data_types::f32, format::bfyx
#define CASE_POOLING_F32_5 { 1, 32, 10, 10 }, data_types::f32, format::byxf, data_types::f32, format::bfyx
#define CASE_POOLING_F32_6 { 1, 32, 40, 40 }, data_types::f32, format::byxf, data_types::f32, format::bfyx
#define CASE_POOLING_F32_7 { 16, 32, 10, 10 }, data_types::f32, format::bs_fs_yx_bsv16_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F32_8 { 16, 32, 10, 10 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F32_9 { 16, 32, 10, 10 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F32_10 { 16, 32, 10, 10, 10 }, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F32_11 { 1, 1, 3, 3 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx

#define CASE_POOLING_F32_F16_1 { 1, 16, 8, 8 }, data_types::f32, format::bfyx, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_2 { 2, 16, 8, 8 }, data_types::f32, format::bfyx, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_3 { 1, 32, 10, 10 }, data_types::f32, format::bfyx, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_4 { 1, 32, 10, 10 }, data_types::f32, format::fs_b_yx_fsv32, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_5 { 1, 32, 10, 10 }, data_types::f32, format::byxf, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_6 { 1, 32, 40, 40 }, data_types::f32, format::byxf, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_7 { 16, 32, 10, 10 }, data_types::f32, format::bs_fs_yx_bsv16_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_8 { 16, 32, 10, 10 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_9 { 16, 32, 10, 10 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_10 { 16, 32, 10, 10, 10 }, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, data_types::f16, format::bfyx

#define CASE_POOLING_F16_1 { 1, 16, 8, 8 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_F16_3 { 1, 32, 10, 10 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_F16_4 { 1, 32, 10, 10 }, data_types::f16, format::fs_b_yx_fsv32, data_types::f32, format::bfyx
#define CASE_POOLING_F16_5 { 1, 32, 10, 10 }, data_types::f16, format::byxf, data_types::f32, format::bfyx
#define CASE_POOLING_F16_6 { 1, 32, 40, 40 }, data_types::f16, format::byxf, data_types::f32, format::bfyx
#define CASE_POOLING_F16_7 { 16, 32, 10, 10 }, data_types::f16, format::bs_fs_yx_bsv16_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F16_8 { 16, 32, 10, 10 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F16_9 { 16, 32, 10, 10, 10 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F16_10 { 16, 32, 10, 10, 10 }, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F16_11 { 1, 64, 10, 10 }, data_types::f16, format::fs_b_yx_fsv32, data_types::f32, format::bfyx

#define CASE_POOLING_F16_FP16_1 { 1, 32, 10, 10 }, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_2 { 1, 32, 10, 10 }, data_types::f16, format::fs_b_yx_fsv32, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_3 { 1, 32, 10, 10 }, data_types::f16, format::byxf, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_4 { 1, 32, 40, 40 }, data_types::f16, format::byxf, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_5 { 16, 32, 10, 10 }, data_types::f16, format::bs_fs_yx_bsv16_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_6 { 16, 32, 10, 10 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_7 { 16, 32, 10, 10, 10 }, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_8 { 16, 32, 10, 10, 10 }, data_types::f16, format::bs_fs_zyx_bsv16_fsv16, data_types::f16, format::bfyx

#define CASE_POOLING_U8_1 { 1, 16, 8, 8 }, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_U8_2 { 2, 16, 8, 8 }, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_U8_3 { 1, 32, 10, 10 }, data_types::u8, format::b_fs_yx_fsv4, data_types::f32, format::b_fs_yx_fsv4
#define CASE_POOLING_U8_5 { 16, 32, 10, 10, 10 }, data_types::u8, format::b_fs_zyx_fsv32, data_types::f32, format::bfyx
#define CASE_POOLING_U8_6 { 16, 32, 10, 10, 10 }, data_types::u8, format::b_fs_zyx_fsv32, data_types::f32, format::bfyx

#define CASE_POOLING_U8_FP16_3 { 1, 32, 10, 10 }, data_types::u8, format::b_fs_yx_fsv4, data_types::f16, format::b_fs_yx_fsv4
#define CASE_POOLING_U8_FP16_5 { 16, 32, 10, 10, 10 }, data_types::u8, format::b_fs_zyx_fsv32, data_types::f16, format::bfyx
#define CASE_POOLING_U8_FP16_6 { 16, 32, 10, 10, 10 }, data_types::u8, format::b_fs_zyx_fsv32, data_types::f16, format::bfyx

#define CASE_POOLING_I8_1 { 1, 16, 8, 8 }, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_I8_2 { 2, 16, 8, 8 }, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_I8_5 { 1, 32, 10, 10 }, data_types::i8, format::b_fs_yx_fsv4, data_types::f32, format::b_fs_yx_fsv4
#define CASE_POOLING_I8_6 { 16, 32, 10, 10, 10 }, data_types::i8, format::b_fs_zyx_fsv32, data_types::f32, format::bfyx

#define CASE_POOLING_I8_FP16_5 { 1, 32, 10, 10 }, data_types::i8, format::b_fs_yx_fsv4, data_types::f16, format::b_fs_yx_fsv4
#define CASE_POOLING_I8_FP16_6 { 16, 32, 10, 10, 10 }, data_types::i8, format::b_fs_zyx_fsv32, data_types::f16, format::bfyx

class pooling_f32_activation : public PoolingFusingTest {};
TEST_P(pooling_f32_activation, basic) {
    auto p = GetParam();

    auto r = get_input_layout(p).get_spatial_rank();
    ov::Shape kernel(r, 3);
    ov::Strides stride(r, 1);
    ov::Shape pad(r, 1);

    create_topologies(
        input_layout("input", get_input_layout(p)),
        pooling("pooling", "input", p.pool_mode, kernel, stride, pad),
        activation("act", "pooling", activation_func::relu),
        reorder("output_reorder", "act", format::bfyx, data_types::f32)
    );

    tolerance = 1e-05f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, pooling_f32_activation, ::testing::ValuesIn(std::vector<pooling_test_params>{
    pooling_test_params{ CASE_POOLING_F32_1, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_F32_1, 2, 3, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_F16_1, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_F16_1, 2, 3, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_I8_1, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_I8_1, 2, 3, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_U8_1, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_U8_1, 2, 3, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_U8_2, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_U8_2, 2, 3, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_I8_1, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_I8_1, 2, 3, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_I8_2, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_I8_2, 2, 3, pooling_mode::average, "" },
}));

class pooling_f32_scale : public PoolingFusingTest {};
TEST_P(pooling_f32_scale, basic) {
    auto p = GetParam();

    auto r = get_input_layout(p).get_spatial_rank();
    ov::Shape kernel(r, 3);
    ov::Strides stride(r, 1);
    ov::Shape pad(r, 1);

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 9.0f)),
        pooling("pooling", "input", p.pool_mode, kernel, stride, pad),
        eltwise("scale", { "pooling", "scale_data" }, eltwise_mode::prod, p.default_type),
        reorder("output_reorder", "scale", format::bfyx, data_types::f32)
    );

    tolerance = 1e-05f;
    execute(p);
}

TEST_P(pooling_f32_scale, fp16_scale_out) {
    auto p = GetParam();

    auto r = get_input_layout(p).get_spatial_rank();
    ov::Shape kernel(r, 3);
    ov::Strides stride(r, 1);
    ov::Shape pad(r, 1);

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 9.0f)),
        pooling("pooling", "input", p.pool_mode, kernel, stride, pad),
        eltwise("scale", { "pooling", "scale_data" }, eltwise_mode::prod, data_types::f16),
        reorder("output_reorder", "scale", format::bfyx, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, pooling_f32_scale, ::testing::ValuesIn(std::vector<pooling_test_params>{
    pooling_test_params{ CASE_POOLING_F32_1, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_F32_1, 2, 3, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_F16_1, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_F16_1, 2, 3, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_U8_1, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_U8_1, 2, 3, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_U8_2, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_U8_2, 2, 3, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_I8_1, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_I8_1, 2, 3, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_I8_2, 2, 3, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_I8_2, 2, 3, pooling_mode::average, "" },
}));

class pooling_scale_activation_quantize : public PoolingFusingTest {};
TEST_P(pooling_scale_activation_quantize, basic) {
    auto p = GetParam();

    auto r = get_input_layout(p).get_spatial_rank();
    ov::Shape kernel(r, 4);
    ov::Strides stride(r, 2);
    ov::Shape pad(r, 0);

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 16.0f)),
        pooling("pooling", "input", "", p.pool_mode, kernel, stride, pad),
        eltwise("scale", { "pooling", "scale_data" }, eltwise_mode::prod, p.default_type),
        activation("activation", "scale", activation_func::relu),
        quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::u8),
        reorder("output_reorder", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

TEST_P(pooling_scale_activation_quantize, i8_output_data_type) {
    auto p = GetParam();

    auto r = get_input_layout(p).get_spatial_rank();
    ov::Shape kernel(r, 4);
    ov::Strides stride(r, 2);
    ov::Shape pad(r, 0);

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127, 127)),
        data("out_hi", get_mem(get_single_element_layout(p), -127, 127)),
        data("scale_data",  get_mem(get_per_channel_layout(p), 1.0f / 16.0f)),
        pooling("pooling", "input", "", p.pool_mode, kernel, stride, pad),
        eltwise("scale", { "pooling", "scale_data" }, eltwise_mode::prod, p.default_type),
        activation("activation", "scale", activation_func::relu),
        quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("output_reorder", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

TEST_P(pooling_scale_activation_quantize, per_channel) {
    auto p = GetParam();

    auto r = get_input_layout(p).get_spatial_rank();
    ov::Shape kernel(r, 4);
    ov::Strides stride(r, 2);
    ov::Shape pad(r, 0);

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 16.0f)),
        pooling("pooling", "input", "", p.pool_mode, kernel, stride, pad),
        eltwise("scale", { "pooling", "scale_data" }, eltwise_mode::prod, p.default_type),
        activation("activation", "scale", activation_func::hyperbolic_tan),
        quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::u8),
        reorder("output_reorder", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, pooling_scale_activation_quantize, ::testing::ValuesIn(std::vector<pooling_test_params>{
    // Input type: FP32
    pooling_test_params{ CASE_POOLING_F32_3, 2, 5, pooling_mode::average, "pooling_gpu_bfyx_block_opt" },
    pooling_test_params{ CASE_POOLING_F32_3, 2, 5, pooling_mode::max, "pooling_gpu_bfyx_block_opt" },
    pooling_test_params{ CASE_POOLING_F32_3, 2, 5, pooling_mode::average, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_3, 2, 5, pooling_mode::max, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_4, 2, 5, pooling_mode::average, "pooling_gpu_fs_b_yx_fsv32" },
    pooling_test_params{ CASE_POOLING_F32_4, 2, 5, pooling_mode::max, "pooling_gpu_fs_b_yx_fsv32" },
    pooling_test_params{ CASE_POOLING_F32_5, 2, 5, pooling_mode::average, "pooling_gpu_byxf_padding_opt" },
    pooling_test_params{ CASE_POOLING_F32_5, 2, 5, pooling_mode::max, "pooling_gpu_byxf_padding_opt" },
    pooling_test_params{ CASE_POOLING_F32_6, 2, 5, pooling_mode::average, "pooling_gpu_byxf_opt" },
    pooling_test_params{ CASE_POOLING_F32_6, 2, 5, pooling_mode::max, "pooling_gpu_byxf_opt" },
    pooling_test_params{ CASE_POOLING_F32_7, 2, 5, pooling_mode::average, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F32_7, 2, 5, pooling_mode::max, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F32_8, 2, 5, pooling_mode::average, "pooling_gpu_blocked" },
    pooling_test_params{ CASE_POOLING_F32_8, 2, 5, pooling_mode::max, "pooling_gpu_blocked" },
    pooling_test_params{ CASE_POOLING_F32_9, 2, 5, pooling_mode::average, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_9, 2, 5, pooling_mode::max, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_10, 2, 5, pooling_mode::average, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F32_10, 2, 5, pooling_mode::max, "pooling_gpu_bsv16_fsv16" },

    // Input type: INT8
    pooling_test_params{ CASE_POOLING_I8_5, 2, 5, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_I8_5, 2, 5, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_I8_6, 2, 5, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_I8_6, 2, 5, pooling_mode::max, "pooling_gpu_int8_ref" },

    // Input type: UINT8
    pooling_test_params{ CASE_POOLING_U8_3, 2, 5, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_3, 2, 5, pooling_mode::max, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_3, 2, 5, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_U8_3, 2, 5, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_U8_5, 2, 5, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_5, 2, 5, pooling_mode::max, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_6, 2, 5, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_6, 2, 5, pooling_mode::max, "pooling_gpu_int8_ref" },
}));

INSTANTIATE_TEST_SUITE_P(DISABLED_fusings_gpu, pooling_scale_activation_quantize, ::testing::ValuesIn(std::vector<pooling_test_params>{
    pooling_test_params{ CASE_POOLING_F32_3, 2, 5, pooling_mode::average, "pooling_gpu_average_opt" },  //currently not enabled, fusing not upported
}));

class pooling_scale_activation : public PoolingFusingTest {};
TEST_P(pooling_scale_activation, basic) {
    auto p = GetParam();

    auto r = get_input_layout(p).get_spatial_rank();
    ov::Shape kernel(r, 4);
    ov::Strides stride(r, 2);
    ov::Shape pad(r, 0);

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 16.0f)),
        pooling("pooling", "input", "", p.pool_mode, kernel, stride, pad),
        eltwise("scale", { "pooling", "scale_data" }, eltwise_mode::prod, p.default_type),
        activation("activation", "scale", activation_func::relu),
        reorder("output_reorder", "activation", p.default_format, data_types::f32)
    );

    tolerance = 1e-05f;
    execute(p);
}

TEST_P(pooling_scale_activation, eltwise_mul) {
    auto p = GetParam();

    auto r = get_input_layout(p).get_spatial_rank();
    ov::Shape kernel(r, 4);
    ov::Strides stride(r, 2);
    ov::Shape pad(r, 0);

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_per_channel_layout(p))),
        pooling("pooling", "input", "", p.pool_mode, kernel, stride, pad),
        eltwise("scale", { "pooling", "scale_data" }, eltwise_mode::prod, p.default_type),
        activation("activation", "scale", activation_func::relu),
        reorder("output_reorder", "activation", p.default_format, data_types::f32)
    );

    tolerance = 1e-05f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, pooling_scale_activation, ::testing::ValuesIn(std::vector<pooling_test_params>{
    // Input type: F32
    pooling_test_params{ CASE_POOLING_F32_3, 2, 4, pooling_mode::average, "pooling_gpu_bfyx_block_opt" },
    pooling_test_params{ CASE_POOLING_F32_3, 2, 4, pooling_mode::max, "pooling_gpu_bfyx_block_opt" },
    pooling_test_params{ CASE_POOLING_F32_3, 2, 4, pooling_mode::average, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_3, 2, 4, pooling_mode::max, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_4, 2, 4, pooling_mode::average, "pooling_gpu_fs_b_yx_fsv32" },
    pooling_test_params{ CASE_POOLING_F32_4, 2, 4, pooling_mode::max, "pooling_gpu_fs_b_yx_fsv32" },
    pooling_test_params{ CASE_POOLING_F32_5, 2, 4, pooling_mode::average, "pooling_gpu_byxf_padding_opt" },
    pooling_test_params{ CASE_POOLING_F32_5, 2, 4, pooling_mode::max, "pooling_gpu_byxf_padding_opt" },
    pooling_test_params{ CASE_POOLING_F32_6, 2, 4, pooling_mode::average, "pooling_gpu_byxf_opt" },
    pooling_test_params{ CASE_POOLING_F32_6, 2, 4, pooling_mode::max, "pooling_gpu_byxf_opt" },
    pooling_test_params{ CASE_POOLING_F32_7, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F32_7, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F32_8, 2, 4, pooling_mode::average, "pooling_gpu_blocked" },
    pooling_test_params{ CASE_POOLING_F32_8, 2, 4, pooling_mode::max, "pooling_gpu_blocked" },
    pooling_test_params{ CASE_POOLING_F32_9, 2, 4, pooling_mode::average, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_9, 2, 4, pooling_mode::max, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_10, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F32_10, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16" },

    // Input type: INT8
    pooling_test_params{ CASE_POOLING_I8_5, 2, 4, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_I8_5, 2, 4, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_I8_6, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_I8_6, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref" },

    // Input type: UINT8
    pooling_test_params{ CASE_POOLING_U8_3, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_3, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_3, 2, 4, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_U8_3, 2, 4, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_U8_5, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_5, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_6, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_6, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref" },

    // Input type: FP16  Output type: F32
    pooling_test_params{ CASE_POOLING_F16_3, 2, 4, pooling_mode::average, "pooling_gpu_bfyx_block_opt" },
    pooling_test_params{ CASE_POOLING_F16_3, 2, 4, pooling_mode::max, "pooling_gpu_bfyx_block_opt" },
    pooling_test_params{ CASE_POOLING_F16_3, 2, 4, pooling_mode::average, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F16_3, 2, 4, pooling_mode::max, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F16_4, 2, 4, pooling_mode::average, "pooling_gpu_fs_b_yx_fsv32" },
    pooling_test_params{ CASE_POOLING_F16_4, 2, 4, pooling_mode::max, "pooling_gpu_fs_b_yx_fsv32" },
    pooling_test_params{ CASE_POOLING_F16_5, 2, 4, pooling_mode::average, "pooling_gpu_byxf_padding_opt" },
    pooling_test_params{ CASE_POOLING_F16_5, 2, 4, pooling_mode::max, "pooling_gpu_byxf_padding_opt" },
    pooling_test_params{ CASE_POOLING_F16_6, 2, 4, pooling_mode::average, "pooling_gpu_byxf_opt" },
    pooling_test_params{ CASE_POOLING_F16_6, 2, 4, pooling_mode::max, "pooling_gpu_byxf_opt" },
    pooling_test_params{ CASE_POOLING_F16_7, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F16_7, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F16_8, 2, 4, pooling_mode::average, "pooling_gpu_blocked" },
    pooling_test_params{ CASE_POOLING_F16_8, 2, 4, pooling_mode::max, "pooling_gpu_blocked" },
    pooling_test_params{ CASE_POOLING_F16_9, 2, 4, pooling_mode::average, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F16_9, 2, 4, pooling_mode::max, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F16_10, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F16_10, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F16_11, 2, 4, pooling_mode::max, "pooling_gpu_fs_b_yx_fsv32" },

    // Input type: FP16
    pooling_test_params{ CASE_POOLING_F16_FP16_1, 2, 4, pooling_mode::average, "pooling_gpu_bfyx_block_opt" },
    pooling_test_params{ CASE_POOLING_F16_FP16_1, 2, 4, pooling_mode::max, "pooling_gpu_bfyx_block_opt" },
    pooling_test_params{ CASE_POOLING_F16_FP16_1, 2, 4, pooling_mode::average, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F16_FP16_1, 2, 4, pooling_mode::max, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F16_FP16_2, 2, 4, pooling_mode::average, "pooling_gpu_fs_b_yx_fsv32" },
    pooling_test_params{ CASE_POOLING_F16_FP16_2, 2, 4, pooling_mode::max, "pooling_gpu_fs_b_yx_fsv32" },
    pooling_test_params{ CASE_POOLING_F16_FP16_3, 2, 4, pooling_mode::average, "pooling_gpu_byxf_padding_opt" },
    pooling_test_params{ CASE_POOLING_F16_FP16_3, 2, 4, pooling_mode::max, "pooling_gpu_byxf_padding_opt" },
    pooling_test_params{ CASE_POOLING_F16_FP16_4, 2, 4, pooling_mode::average, "pooling_gpu_byxf_opt" },
    pooling_test_params{ CASE_POOLING_F16_FP16_4, 2, 4, pooling_mode::max, "pooling_gpu_byxf_opt" },
    pooling_test_params{ CASE_POOLING_F16_FP16_5, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F16_FP16_5, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F16_FP16_6, 2, 4, pooling_mode::average, "pooling_gpu_blocked" },
    pooling_test_params{ CASE_POOLING_F16_FP16_6, 2, 4, pooling_mode::max, "pooling_gpu_blocked" },
    pooling_test_params{ CASE_POOLING_F16_FP16_7, 2, 4, pooling_mode::average, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F16_FP16_7, 2, 4, pooling_mode::max, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F16_FP16_8, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F16_FP16_8, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16" },

    // Input type: FP32
    pooling_test_params{ CASE_POOLING_F32_F16_3, 2, 4, pooling_mode::average, "pooling_gpu_bfyx_block_opt" },
    pooling_test_params{ CASE_POOLING_F32_F16_3, 2, 4, pooling_mode::max, "pooling_gpu_bfyx_block_opt" },
    pooling_test_params{ CASE_POOLING_F32_F16_3, 2, 4, pooling_mode::average, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_F16_3, 2, 4, pooling_mode::max, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_F16_4, 2, 4, pooling_mode::average, "pooling_gpu_fs_b_yx_fsv32" },
    pooling_test_params{ CASE_POOLING_F32_F16_4, 2, 4, pooling_mode::max, "pooling_gpu_fs_b_yx_fsv32" },
    pooling_test_params{ CASE_POOLING_F32_F16_5, 2, 4, pooling_mode::average, "pooling_gpu_byxf_padding_opt" },
    pooling_test_params{ CASE_POOLING_F32_F16_5, 2, 4, pooling_mode::max, "pooling_gpu_byxf_padding_opt" },
    pooling_test_params{ CASE_POOLING_F32_F16_6, 2, 4, pooling_mode::average, "pooling_gpu_byxf_opt" },
    pooling_test_params{ CASE_POOLING_F32_F16_6, 2, 4, pooling_mode::max, "pooling_gpu_byxf_opt" },
    pooling_test_params{ CASE_POOLING_F32_F16_7, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F32_F16_7, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F32_F16_8, 2, 4, pooling_mode::average, "pooling_gpu_blocked" },
    pooling_test_params{ CASE_POOLING_F32_F16_8, 2, 4, pooling_mode::max, "pooling_gpu_blocked" },
    pooling_test_params{ CASE_POOLING_F32_F16_9, 2, 4, pooling_mode::average, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_F16_9, 2, 4, pooling_mode::max, "pooling_gpu_ref" },
    pooling_test_params{ CASE_POOLING_F32_F16_10, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16" },
    pooling_test_params{ CASE_POOLING_F32_F16_10, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16" },

    // Input type: INT8
    pooling_test_params{ CASE_POOLING_I8_FP16_5, 2, 4, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_I8_FP16_5, 2, 4, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_I8_FP16_6, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_I8_FP16_6, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref" },

    // Input type: UINT8
    pooling_test_params{ CASE_POOLING_U8_FP16_3, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_FP16_3, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_FP16_3, 2, 4, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_U8_FP16_3, 2, 4, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4" },
    pooling_test_params{ CASE_POOLING_U8_FP16_5, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_FP16_5, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_FP16_6, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref" },
    pooling_test_params{ CASE_POOLING_U8_FP16_6, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref" },
}));

#ifdef ENABLE_ONEDNN_FOR_GPU
class PoolingOneDNNFusingTest : public ::BaseFusingTest<pooling_test_params> {
public:
    void execute(pooling_test_params& p) {
        // Onednn post operation has issue in a machine that does not support imad.
        if (!engine.get_device_info().supports_immad)
            return;

        auto input_prim = get_mem(get_input_layout(p));

        build_options onednn_options;
        build_options cldnn_options;

        onednn_options.set_option(build_option::optimize_data(true));
        cldnn_options.set_option(build_option::optimize_data(true));

        implementation_desc onednn_impl = { p.input_format, "", impl_types::onednn };
        implementation_desc cldnn_impl = { p.input_format, "", impl_types::ocl };
        onednn_options.set_option(build_option::force_implementations({ { "pooling", onednn_impl } }));
        cldnn_options.set_option(build_option::force_implementations({ { "pooling", cldnn_impl } }));

        // for onednn fusing test, topology_non_fused means cldnn, topology_fused is onednn
        network network_fused_cldnn(this->engine, this->topology_non_fused, cldnn_options);
        network network_fused_onednn(this->engine, this->topology_fused, onednn_options);

        network_fused_cldnn.set_input_data("input", input_prim);
        network_fused_onednn.set_input_data("input", input_prim);

        ASSERT_FALSE(network_fused_cldnn.get_primitives_info().empty());
        ASSERT_FALSE(network_fused_onednn.get_primitives_info().empty());

        auto find_and_check = [&](primitive_info& p) -> bool {
            if (p.original_id == "pooling" || p.original_id == "output_reorder")
                return true;
            return false;
        };

        auto pi_fused_onednn = network_fused_onednn.get_primitives_info();
        auto pi_fused_cldnn = network_fused_onednn.get_primitives_info();
        auto info_fused_onednn = std::find_if(pi_fused_onednn.begin(), pi_fused_onednn.end(), find_and_check);
        auto info_fused_cldnn = std::find_if(pi_fused_cldnn.begin(), pi_fused_cldnn.end(), find_and_check);

        ASSERT_TRUE(info_fused_onednn != pi_fused_onednn.end());
        ASSERT_TRUE(info_fused_cldnn != pi_fused_cldnn.end());

        compare(network_fused_cldnn, network_fused_onednn, p);
    }

    layout get_input_layout(pooling_test_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape };
    }

    layout get_per_channel_layout(pooling_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.in_shape.feature[0], 1, 1 } };
    }
};

class pooling_onednn_activation1 : public PoolingOneDNNFusingTest {};
TEST_P(pooling_onednn_activation1, basic) {
    auto p = GetParam();

    auto r = get_input_layout(p).get_spatial_rank();
    ov::Shape kernel(r, 3);
    ov::Strides stride(r, 1);
    ov::Shape pad(r, 1);

    create_topologies(
        input_layout("input", get_input_layout(p)),
        pooling("pooling", "input", p.pool_mode, kernel, stride, pad),
        activation("act", "pooling", activation_func::relu),
        reorder("output_reorder", "act", format::bfyx, data_types::f32)
    );

    tolerance = 1e-05f;
    execute(p);
}

class pooling_onednn_activation2 : public PoolingOneDNNFusingTest {};
TEST_P(pooling_onednn_activation2, basic) {
    auto p = GetParam();

    auto r = get_input_layout(p).get_spatial_rank();
    ov::Shape kernel(r, 3);
    ov::Strides stride(r, 1);

    create_topologies(
        input_layout("input", get_input_layout(p)),
        pooling("pooling", "input", p.pool_mode, kernel, stride),
        activation("act", "pooling", activation_func::relu),
        reorder("output_reorder", "act", format::bfyx, data_types::f32)
    );

    tolerance = 1e-05f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, pooling_onednn_activation1, ::testing::ValuesIn(std::vector<pooling_test_params>{
    // pooling_test_params{ CASE_POOLING_F32_1, 2, 2, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_F16_1, 2, 2, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_I8_1, 2, 2, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_U8_1, 2, 2, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_U8_2, 2, 2, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_I8_1, 2, 2, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_I8_2, 2, 2, pooling_mode::max, "" },
}));

INSTANTIATE_TEST_SUITE_P(fusings_gpu, pooling_onednn_activation2, ::testing::ValuesIn(std::vector<pooling_test_params>{
    pooling_test_params{ CASE_POOLING_F32_11, 2, 2, pooling_mode::max, "" },
    pooling_test_params{ CASE_POOLING_F32_11, 2, 2, pooling_mode::average, "" },
    pooling_test_params{ CASE_POOLING_F32_11, 2, 2, pooling_mode::average_no_padding, "" },
}));
#endif
