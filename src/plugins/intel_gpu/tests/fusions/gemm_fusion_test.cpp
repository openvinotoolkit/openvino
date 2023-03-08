// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/runtime/tensor.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::details;
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
    std::string kernel_name;
    dim_vec_kind broadcast_kind;
    eltwise_mode eltwise_m;
};

class GemmFusingTest : public ::BaseFusingTest<gemm_test_params> {
public:

    void execute(gemm_test_params& p) {
        auto input0_prim = get_mem(get_input_layout(p, 0));
        auto input1_prim = get_mem(get_input_layout(p, 1));

        if (!p.kernel_name.empty()) {
            ov::intel_gpu::ImplementationDesc gemm_ref_impl = { format::bfyx, "gemm_ref" };
            ov::intel_gpu::ImplementationDesc gemm_target_impl = { format::bfyx, p.kernel_name };
            cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gemm_prim", gemm_target_impl} }));
            cfg_not_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gemm_prim", gemm_ref_impl} }));
        }

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
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
#define CASE_GEMM_2IN_FP16_5 { { 2, 3, 2, 2 }, { 2, 3, 2, 2 } }, { 2, 3, 2, 2 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_U8U8_1 { { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, { 1, 1, 2, 2 }, tensor{ 1 }, tensor{ 0 }, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_U8U8_2 { { 1, 2, 64, 128 }, { 1, 2, 256, 64 } }, { 1, 2, 256, 128 }, tensor{ 1 }, tensor{ 0 }, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_U8U8_3 { { 1, 1, 16, 32 }, { 1, 1, 32, 16 } }, { 1, 1, 32, 32 }, tensor{ 1 }, tensor{ 0 }, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_2IN_U8S8_1 { { 1, 1, 4, 2 }, { 1, 1, 8, 4 } }, { 1, 1, 8, 4 }, tensor{ 1 }, tensor{ 0 }, data_types::u8, data_types::i8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_S8U8_1 { { 1, 2, 64, 128 }, { 1, 2, 256, 64 } }, { 1, 2, 256, 128 }, tensor{ 1 }, tensor{ 0 }, data_types::i8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_ELTWISE_2IN_FP32_1 { { 1, 1, 4, 4 }, { 1, 1, 4, 4 } }, { 1, 1, 4, 4 }, tensor{ 1 }, tensor{ 0 }, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_FP16_1 { { 1, 1, 32, 32 }, { 1, 1, 32, 32 } }, { 1, 1, 32, 32 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_FP16_2 { { 1, 1, 1024, 1024 }, { 1, 1, 1024, 1024 } }, { 1, 1, 1024, 1024 }, tensor{ 1 }, tensor{ 0 }, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_U8S8_1 { { 1, 1, 4, 4 }, { 1, 1, 4, 4 } }, { 1, 1, 4, 4 }, tensor{ 1 }, tensor{ 0 }, data_types::u8, data_types::i8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_S8U8_1 { { 1, 1, 32, 32 }, { 1, 1, 32, 32 } }, { 1, 1, 32, 32 }, tensor{ 1 }, tensor{ 0 }, data_types::i8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_U8S8_2 { { 1, 1, 1024, 1024 }, { 1, 1, 1024, 1024 } }, { 1, 1, 1024, 1024 }, tensor{ 1 }, tensor{ 0 }, data_types::u8, data_types::i8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

class gemm_3in_quantize_i8 : public GemmFusingTest {};
TEST_P(gemm_3in_quantize_i8, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        input_layout("input2", get_input_layout(p, 2)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1"), input_info("input2") }, data_types::f32),
        quantize("quantize", input_info("gemm_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(data_types::i8);
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
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        quantize("quantize", input_info("gemm_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(data_types::u8);
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
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        quantize("quantize", input_info("gemm_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc gemm_impl = { format::bfyx, "gemm_tiled_opt" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm_prim", gemm_impl } }));

    tolerance = default_tolerance(data_types::u8);
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
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        eltwise("scale", { input_info("gemm_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

TEST_P(gemm_2in_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        eltwise("scale", { input_info("gemm_prim"), input_info("scale_data") }, eltwise_mode::prod, data_types::f16),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
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


class gemm_2in_add : public GemmFusingTest {};
TEST_P(gemm_2in_add, eltwise_postop) {
    auto p = GetParam();

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc gemmv_impl = { cldnn::format::type::any, "", impl_types::onednn };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm_prim", gemmv_impl } }));
        cfg_fused.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    }

    auto add_data_layout = get_output_layout(p);
    auto add_data_size = add_data_layout.get_tensor();
    if (p.broadcast_kind == dim_vec_kind::batch)
        add_data_size.batch[0] = 1;
    else
        add_data_size.feature[0] = 1;
    add_data_layout.set_tensor(add_data_size);

    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("add_data", get_mem(add_data_layout, 1.0f/p.kernel.count())),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        eltwise("add_prim", { input_info("gemm_prim"), input_info("add_data") }, p.eltwise_m, p.default_type),
        reorder("reorder_bfyx", input_info("add_prim"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_add, ::testing::ValuesIn(std::vector<gemm_test_params>{
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", dim_vec_kind::batch, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", dim_vec_kind::batch, eltwise_mode::prod },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", dim_vec_kind::batch, eltwise_mode::sub },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", dim_vec_kind::feature, eltwise_mode::sum },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", dim_vec_kind::feature, eltwise_mode::prod },
    gemm_test_params{ CASE_GEMM_2IN_FP16_5, 3, 4, "", dim_vec_kind::feature, eltwise_mode::sub },
}));

class gemm_2in_act_scale_quantize_i8 : public GemmFusingTest {};
TEST_P(gemm_2in_act_scale_quantize_i8, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        activation("activation", input_info("gemm_prim"), activation_func::exp),
        eltwise("scale", { input_info("activation"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        quantize("quantize", input_info("scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(data_types::i8);
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
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        activation("activation", input_info("gemm_prim"), activation_func::exp),
        eltwise("scale", { input_info("activation"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        quantize("quantize", input_info("scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(data_types::i8);
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
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        eltwise("scale", { input_info("gemm_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        activation("activation", input_info("scale"), activation_func::negative),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

TEST_P(gemm_2in_act_scale_eltwise, broadcast_eltwise) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        data("eltwise_data", get_mem(get_single_element_layout(p))),
        gemm("gemm_prim", { input_info("input0"), input_info("input1") }, data_types::f32),
        eltwise("scale", { input_info("gemm_prim"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        activation("activation", input_info("scale"), activation_func::negative),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    gemm_2in_act_scale_eltwise,
    ::testing::ValuesIn(std::vector<gemm_test_params>{
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_FP32_1, 3, 6},
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_FP16_1, 3, 6},
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_U8S8_1, 3, 6},
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_S8U8_1, 3, 6},
        // Reference graph can be fused because force_implementation leads optimize_data(true) in program::set_options()
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_U8S8_2, 3, 3, "gemm_mmad_int8"},
        // gemm_test_params{ CASE_GEMM_ELTWISE_2IN_U8S8_2, 3, 3, "gemm_mmad_int8_slm" },   // tolerance issue
        gemm_test_params{CASE_GEMM_ELTWISE_2IN_FP16_2, 3, 3, "gemm_tiled_opt"},
    }));
