// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct gemm_test_params {
    std::vector<tensor> in_shapes;
    tensor out_shape;
    tensor kernel;
    tensor pad;
    data_types data_type_in0;
    data_types data_type_in1;
    data_types data_type_in2;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class GemmFusingTest : public ::BaseFusingTest<gemm_test_params> {
public:

    void execute(gemm_test_params& p) {
        auto input0_prim = get_mem(get_input_layout(p, 0));
        auto input1_prim = get_mem(get_input_layout(p, 1));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input0", input0_prim);
        network_not_fused.set_input_data("input0", input0_prim);
        network_fused.set_input_data("input1", input1_prim);
        network_not_fused.set_input_data("input1", input1_prim);
        if (p.in_shapes.size() > 2) {
            auto input2_prim = get_mem(get_input_layout(p, 2));
            network_fused.set_input_data("input2", input2_prim);
            network_not_fused.set_input_data("input2", input2_prim);
        }

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(gemm_test_params& p, int in_no) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        if (in_no == 0)
            return layout{ p.data_type_in0, p.input_format, p.in_shapes.at(0), padding{ pad_ } };
        else if (in_no == 1)
            return layout{ p.data_type_in1, p.input_format, p.in_shapes.at(1), padding{ pad_ } };
        else
            return layout{ p.data_type_in2, p.input_format, p.in_shapes.at(2), padding{ pad_ } };
    }

    layout get_per_channel_layout(gemm_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.in_shapes.at(0).feature[0], 1, 1 } };
    }

    layout get_output_layout(gemm_test_params& p) {
        return layout{ p.default_type, p.input_format, p.out_shape };
    }
};

}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- Gemm cases ------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_GEMM_3IN_FP32_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, tensor{ 1 }, tensor{ 0 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_FP32_2 { { 1, 1, 63, 63 }, { 1, 1, 63, 63 }, { 1, 1, 63, 63 } }, { 1, 1, 63, 63 }, tensor{ 1 }, tensor{ 0 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_FP32_3 { { 1, 1, 128, 128 }, { 1, 1, 128, 128 }, { 1, 1, 128, 128 } }, { 1, 1, 128, 128 }, tensor{ 1 }, tensor{ 0 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_FP32_4 { { 1, 2, 64, 128 }, { 1, 2, 256, 64 }, { 1, 2, 256, 128 } }, { 1, 2, 256, 128 }, tensor{ 1 }, tensor{ 0 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_FP16_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_3IN_FP16_2 { { 1, 1, 31, 31 }, { 1, 1, 31, 31 }, { 1, 1, 31, 31 } }, { 1, 1, 31, 31 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_3IN_FP16_3 { { 1, 1, 64, 64 }, { 1, 1, 64, 64 }, { 1, 1, 64, 64 } }, { 1, 1, 64, 64 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_3IN_FP16_4 { { 1, 2, 64, 128 }, { 1, 2, 256, 64 }, { 1, 2, 256, 128 } }, { 1, 2, 256, 128 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_3IN_S8S8_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, tensor{ 1 }, tensor{ 0 }, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_S8S8_2 { { 1, 2, 64, 128 }, { 1, 2, 256, 64 }, { 1, 2, 256, 128 } }, { 1, 2, 256, 128 }, tensor{ 1 }, tensor{ 0 }, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_S8S8_3 { { 1, 1, 8, 16 }, { 1, 1, 32, 8 }, { 1, 1, 32, 16 } }, { 1, 1, 32, 16 }, tensor{ 1 }, tensor{ 0 }, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_2IN_FP32_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, tensor{ 1 }, tensor{ 0 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_FP32_2 { { 1, 1, 63, 63 }, { 1, 1, 63, 63 } }, { 1, 1, 63, 63 }, tensor{ 1 }, tensor{ 0 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_FP32_3 { { 1, 1, 128, 128 }, { 1, 1, 128, 128 } }, { 1, 1, 128, 128 }, tensor{ 1 }, tensor{ 0 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_FP32_4 { { 1, 2, 64, 128 }, { 1, 2, 256, 64 } }, { 1, 2, 256, 128 }, tensor{ 1 }, tensor{ 0 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_FP16_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_FP16_2 { { 1, 1, 31, 31 }, { 1, 1, 31, 31 } }, { 1, 1, 31, 31 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_FP16_3 { { 1, 1, 64, 64 }, { 1, 1, 64, 64 } }, { 1, 1, 64, 64 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_FP16_4 { { 1, 2, 64, 128 }, { 1, 2, 256, 64 } }, { 1, 2, 256, 128 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_U8U8_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, tensor{ 1 }, tensor{ 0 }, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_U8U8_2 { { 1, 2, 64, 128 }, { 1, 2, 256, 64 } }, { 1, 2, 256, 128 }, tensor{ 1 }, tensor{ 0 }, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_U8U8_3 { { 1, 1, 16, 32 }, { 1, 1, 32, 16 } }, { 1, 1, 32, 32 }, tensor{ 1 }, tensor{ 0 }, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_2IN_U8S8_1 { { 1, 1, 4, 2 }, { 1, 1, 8, 4 } }, { 1, 1, 8, 4 }, tensor{ 1 }, tensor{ 0 }, data_types::u8, data_types::i8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_S8U8_1 { { 1, 2, 64, 128 }, { 1, 2, 256, 64 } }, { 1, 2, 256, 128 }, tensor{ 1 }, tensor{ 0 }, data_types::i8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_ELTWISE_2IN_FP32_1 { { 1, 1, 4, 4 }, { 1, 1, 4, 4 } }, { 1, 1, 4, 4 }, tensor{ 1 }, tensor{ 0 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_FP16_1 { { 1, 1, 32, 32 }, { 1, 1, 32, 32 } }, { 1, 1, 32, 32 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_U8S8_1 { { 1, 1, 4, 4 }, { 1, 1, 4, 4 } }, { 1, 1, 4, 4 }, tensor{ 1 }, tensor{ 0 }, data_types::u8, data_types::i8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_S8U8_1 { { 1, 1, 32, 32 }, { 1, 1, 32, 32 } }, { 1, 1, 32, 32 }, tensor{ 1 }, tensor{ 0 }, data_types::i8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

class gemm_3in_quantize_i8 : public GemmFusingTest {};
TEST_P(gemm_3in_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        input_layout("input2", get_input_layout(p, 2)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gemm("gemm_prim", { "input0", "input1", "input2" }, data_types::f32),
        quantize("quantize", "gemm_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_3in_quantize_i8, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_3IN_FP16_1, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP16_2, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP16_3, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP16_4, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP32_1, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP32_2, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP32_3, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_FP32_4, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_S8S8_1, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_S8S8_2, 4, 5 },
    gemm_test_params{ CASE_GEMM_3IN_S8S8_3, 4, 5 },
}));

class gemm_2in_quantize_u8 : public GemmFusingTest {};
TEST_P(gemm_2in_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        quantize("quantize", "gemm_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_quantize_u8, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_2IN_FP16_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_3, 3, 4 },
}));

class gemm_2in_quantize_float_in : public GemmFusingTest {};
TEST_P(gemm_2in_quantize_float_in, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        quantize("quantize", "gemm_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    implementation_desc gemm_impl = { format::bfyx, "gemm_tiled_opt" };
    bo_fused.set_option(build_option::force_implementations({ { "gemm_prim", gemm_impl } }));

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_quantize_float_in, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_2IN_FP16_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP16_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP32_1, 3, 4 },
}));

class gemm_2in_scale : public GemmFusingTest {};
TEST_P(gemm_2in_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        scale("scale", "gemm_prim", "scale_data"),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(gemm_2in_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        scale("scale", "gemm_prim", "scale_data", optional_data_type{ data_types::f16 }),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_scale, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_2IN_FP32_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_4, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_1, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_2, 3, 4 },
    gemm_test_params{ CASE_GEMM_2IN_U8U8_3, 3, 4 },
}));

class gemm_2in_act_scale_quantize_i8 : public GemmFusingTest {};
TEST_P(gemm_2in_act_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        activation("activation", "gemm_prim", activation_func::exp),
        scale("scale", "activation", "scale_data"),
        quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_act_scale_quantize_i8, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_2IN_FP32_1, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_2, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_3, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP32_4, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_1, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_2, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_3, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_FP16_4, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_U8S8_1, 3, 6 },
    gemm_test_params{ CASE_GEMM_2IN_S8U8_1, 3, 6 },
}));

class gemm_2in_act_scale_quantize_eltwise_i8 : public GemmFusingTest {};
TEST_P(gemm_2in_act_scale_quantize_eltwise_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        activation("activation", "gemm_prim", activation_func::exp),
        scale("scale", "activation", "scale_data"),
        quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        eltwise("sum", { "quantize", "eltwise_data" }, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_act_scale_quantize_eltwise_i8, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP32_1, 3, 7 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP16_1, 3, 7 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_U8S8_1, 3, 7 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_S8U8_1, 3, 7 },
}));

class gemm_2in_act_scale_eltwise : public GemmFusingTest {};
TEST_P(gemm_2in_act_scale_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        scale("scale", "gemm_prim", "scale_data"),
        activation("activation", "scale", activation_func::negative),
        eltwise("sum", { "activation", "eltwise_data" }, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );

    tolerance = 1e-4f;
    execute(p);
}

TEST_P(gemm_2in_act_scale_eltwise, broadcast_eltwise) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        data("eltwise_data", get_mem(get_single_element_layout(p))),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        scale("scale", "gemm_prim", "scale_data"),
        activation("activation", "scale", activation_func::negative),
        eltwise("sum", { "activation", "eltwise_data" }, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );

    tolerance = 1e-4f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_act_scale_eltwise, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP32_1, 3, 6 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP16_1, 3, 6 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_U8S8_1, 3, 6 },
    gemm_test_params{ CASE_GEMM_ELTWISE_2IN_S8U8_1, 3, 6 },
}));
