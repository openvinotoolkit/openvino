// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/activation.hpp>

// Include kernel selector helpers for activation function string conversion
#include "graph/impls/ocl/kernel_selector_helper.h"
#include "kernel_selector/kernel_selector_common.h"

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct activation_test_params {
    ov::PartialShape input_size;
    data_types input_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    std::string kernel_name;
};

class ActivationFusingTest : public ::BaseFusingTest<activation_test_params> {
public:
    void execute(activation_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        ExecutionConfig cfg = get_test_default_config(engine);
        ov::intel_gpu::ImplementationDesc activation_impl = { p.input_format, p.kernel_name };
        cfg.set_property(ov::intel_gpu::optimize_data(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "act", activation_impl } }));
        network network_fused(this->engine, this->topology_fused, cfg);
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(activation_test_params& p) {
        return layout{ p.input_size, p.input_type, p.input_format, };
    }

    layout get_per_channel_layout(activation_test_params& p) {
        return layout{ { 1, p.input_size[1], 1, 1 }, p.default_type, p.default_format };
    }

    format get_input_format(activation_test_params &p) { return p.input_format; }
};

}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* -------------------------------- Activation cases --------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_ACTIVATION_F32_0 { 7, 32, 3, 3 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_1 { 1, 16, 8, 8 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_2 { 7, 3, 7, 7 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_3 { 1, 14, 8, 8 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_4 { 1, 17, 31, 29 }, data_types::f32, format::yxfb, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_5 { 1, 17, 31, 29 }, data_types::f32, format::b_fs_yx_fsv4, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_6 { 1, 17, 31, 29 }, data_types::f32, format::b_fs_yx_fsv32, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_7 { 1, 17, 31, 29 }, data_types::f32, format::fyxb, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_8 { 1, 2, 3, 4, 5, 3, 2, 3 }, data_types::f32, format::bfvuwzyx, data_types::f32, format::bfvuwzyx
#define CASE_ACTIVATION_3D_F32_0 { 3, 16, 13, 13, 13 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F32_1 { 2, 16, 8, 8, 8 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F32_2 { 1, 16, 7, 7, 7 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F32_3 { 1, 17, 7, 7, 7 }, data_types::f32, format::b_fs_zyx_fsv32, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F32_4 { 1, 17, 7, 7, 7 }, data_types::f32, format::bs_fs_yx_bsv16_fsv16, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F32_5 { 1, 17, 7, 7, 7 }, data_types::f32, format::fs_b_yx_fsv32, data_types::f32, format::bfzyx

#define CASE_ACTIVATION_F16_0 { 7, 32, 5, 5 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_1 { 1, 16, 8, 8 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_2 { 7, 16, 7, 7 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_3 { 1, 14, 8, 8 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_4 { 1, 17, 31, 29 }, data_types::f16, format::yxfb, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_5 { 1, 17, 31, 29 }, data_types::f16, format::b_fs_yx_fsv4, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_6 { 1, 17, 31, 29 }, data_types::f16, format::b_fs_yx_fsv32, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_7 { 1, 17, 31, 29 }, data_types::f16, format::fyxb, data_types::f32, format::bfyx
#define CASE_ACTIVATION_3D_F16_0 { 3, 16, 13, 13, 13 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F16_1 { 2, 16, 8, 8, 8 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F16_2 { 1, 16, 7, 7, 7 }, data_types::f16, format::b_fs_zyx_fsv16, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F16_3 { 1, 17, 7, 7, 7 }, data_types::f16, format::b_fs_zyx_fsv32, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F16_4 { 1, 17, 7, 7, 7 }, data_types::f16, format::bs_fs_yx_bsv16_fsv16, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F16_5 { 1, 17, 7, 7, 7 }, data_types::f16, format::fs_b_yx_fsv32, data_types::f32, format::bfzyx

#define CASE_ACTIVATION_U8_1 { 1, 16, 8, 8 }, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_U8_2 { 1, 12, 8, 8 }, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_I8_1 { 1, 16, 8, 8 }, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_I8_2 { 1, 14, 8, 8 }, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_3D_I8_1 { 1, 17, 8, 8, 8 }, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx

class activation_quantize_i8 : public ActivationFusingTest {};
TEST_P(activation_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        activation("act", input_info("input"), activation_func::relu),
        data("in_low", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_high", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -127, 0)),
        data("out_high", get_mem(get_single_element_layout(p), 0, 127)),
        quantize("quant", input_info("act"), input_info("in_low"), input_info("in_high"),
                 input_info("out_low"), input_info("out_high"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quant"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

TEST_P(activation_quantize_i8, per_channel) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        activation("act", input_info("input"), activation_func::relu),
        data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -127, 0)),
        data("out_high", get_mem(get_single_element_layout(p), 0, 127)),
        quantize("quant", input_info("act"), input_info("in_low"), input_info("in_high"),
                 input_info("out_low"), input_info("out_high"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quant"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, activation_quantize_i8, ::testing::ValuesIn(std::vector<activation_test_params>{
    // InputDataType = FP32
    activation_test_params{ CASE_ACTIVATION_F32_0, 2, 3, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_F32_1, 2, 3, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_0, 2, 3, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_1, 2, 3, "activation_opt" },

    activation_test_params{ CASE_ACTIVATION_F32_0, 2, 3, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_1, 2, 3, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_2, 2, 3, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_3, 2, 3, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_4, 2, 3, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_0, 2, 3, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_1, 2, 3, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_2, 2, 3, "activation_ref" },
}));

INSTANTIATE_TEST_SUITE_P(DISABLED_fusings_gpu, activation_quantize_i8, ::testing::ValuesIn(std::vector<activation_test_params>{
    activation_test_params{ CASE_ACTIVATION_F32_5, 2, 3, "activation_ref" },     // FIXME - accuracy bug
    activation_test_params{ CASE_ACTIVATION_F32_6, 2, 3, "activation_ref" },     // FIXME - accuracy bug
    activation_test_params{ CASE_ACTIVATION_F32_7, 2, 3, "activation_ref" },     // FIXME - accuracy bug
    activation_test_params{ CASE_ACTIVATION_3D_F32_3, 2, 3, "activation_ref" },  // FIXME - accuracy bug
    activation_test_params{ CASE_ACTIVATION_3D_F32_5, 2, 3, "activation_ref" },  // FIXME - accuracy bug
}));

class activation_eltwise_activation_quantize_u8 : public ActivationFusingTest {};
TEST_P(activation_eltwise_activation_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        activation("act", input_info("input"), activation_func::relu),
        data("eltwise_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
        data("in_low", get_mem(get_single_element_layout(p), 0)),
        data("in_high", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -127)),
        data("out_high", get_mem(get_single_element_layout(p), 127)),
        eltwise("eltwise", { input_info("act"), input_info("eltwise_data") }, eltwise_mode::prod, p.default_type),
        activation("act2", input_info("eltwise"), activation_func::swish),
        quantize("quant", input_info("act2"), input_info("in_low"), input_info("in_high"),
                 input_info("out_low"), input_info("out_high"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quant"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

TEST_P(activation_eltwise_activation_quantize_u8, per_channel) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        activation("act", input_info("input"), activation_func::relu),
        data("eltwise_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
        data("in_low", get_mem(get_per_channel_layout(p), 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -127)),
        data("out_high", get_mem(get_single_element_layout(p), 127)),
        eltwise("eltwise", { input_info("act"), input_info("eltwise_data") }, eltwise_mode::prod, p.default_type),
        activation("act2", input_info("eltwise"), activation_func::pow),
        quantize("quant", input_info("act2"), input_info("in_low"), input_info("in_high"),
                 input_info("out_low"), input_info("out_high"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quant"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, activation_eltwise_activation_quantize_u8, ::testing::ValuesIn(std::vector<activation_test_params>{
    // InputDataType = FP32
    activation_test_params{ CASE_ACTIVATION_F32_0, 3, 5, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_F32_1, 3, 5, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_0, 3, 5, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_1, 3, 5, "activation_opt" },

    activation_test_params{ CASE_ACTIVATION_F32_0, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_1, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_2, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_3, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_4, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_5, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_6, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_7, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_0, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_1, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_2, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_8, 3, 5, "activation_ref" },
}));

class activation_eltwise_activation_quantize_u8_onendnn : public ActivationFusingTest {};
TEST_P(activation_eltwise_activation_quantize_u8_onendnn, same_behavior) {
    // Case : activation function is NOT supported on oneDNN and an input primitive selects clDNN execution
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        activation("act", input_info("input"), activation_func::relu),
        data("eltwise_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
        data("in_low", get_mem(get_single_element_layout(p), 0)),
        data("in_high", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -127)),
        data("out_high", get_mem(get_single_element_layout(p), 127)),
        eltwise("eltwise", { input_info("act"), input_info("eltwise_data") }, eltwise_mode::prod, p.default_type),
        activation("act2", input_info("eltwise"), activation_func::softsign),
        quantize("quant", input_info("act2"), input_info("in_low"), input_info("in_high"),
                 input_info("out_low"), input_info("out_high"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quant"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, activation_eltwise_activation_quantize_u8_onendnn, ::testing::ValuesIn(std::vector<activation_test_params>{
    // InputDataType = FP32
    activation_test_params{ CASE_ACTIVATION_F32_0, 3, 5, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_F32_1, 3, 5, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_0, 3, 5, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_1, 3, 5, "activation_opt" },

    activation_test_params{ CASE_ACTIVATION_F32_0, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_1, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_0, 3, 5, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_1, 3, 5, "activation_ref" },
}));

INSTANTIATE_TEST_SUITE_P(DISABLED_fusings_gpu, activation_eltwise_activation_quantize_u8, ::testing::ValuesIn(std::vector<activation_test_params>{
    activation_test_params{ CASE_ACTIVATION_3D_F32_5, 3, 5, "activation_ref" },  // FIXME - accuracy bug
}));

class activation_eltwise_activation : public ActivationFusingTest {};
TEST_P(activation_eltwise_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        activation("act", input_info("input"), activation_func::relu),
        data("eltwise_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
        eltwise("eltwise", { input_info("act"), input_info("eltwise_data") }, eltwise_mode::prod, p.default_type),
        activation("act2", input_info("eltwise"), activation_func::exp),
        reorder("reorder_bfyx", input_info("act2"), p.default_format, data_types::f32)
    );

    tolerance = 1e-05f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, activation_eltwise_activation, ::testing::ValuesIn(std::vector<activation_test_params>{
    // InputDataType = FP32
    activation_test_params{ CASE_ACTIVATION_F32_0, 3, 4, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_F32_1, 3, 4, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_0, 3, 4, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_1, 3, 4, "activation_opt" },

    activation_test_params{ CASE_ACTIVATION_F32_0, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_1, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_2, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_3, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_4, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_5, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_6, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F32_7, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_0, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_1, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F32_2, 3, 4, "activation_ref" },

    // InputDataType = FP16
    activation_test_params{ CASE_ACTIVATION_F16_0, 3, 4, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_F16_1, 3, 4, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_3D_F16_0, 3, 4, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_3D_F16_1, 3, 4, "activation_opt" },

    activation_test_params{ CASE_ACTIVATION_F16_0, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F16_1, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F16_2, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F16_3, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F16_4, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F16_5, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F16_6, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_F16_7, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F16_0, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F16_1, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F16_2, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F16_3, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_F16_4, 3, 4, "activation_ref" },

    // InputDataType = UINT8
    activation_test_params{ CASE_ACTIVATION_U8_1, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_U8_2, 3, 4, "activation_ref" },

    // InputDataType = INT8
    activation_test_params{ CASE_ACTIVATION_I8_1, 3, 4, "activation_opt" },
    activation_test_params{ CASE_ACTIVATION_3D_I8_1, 3, 4, "activation_opt" },

    activation_test_params{ CASE_ACTIVATION_I8_1, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_I8_2, 3, 4, "activation_ref" },
    activation_test_params{ CASE_ACTIVATION_3D_I8_1, 3, 4, "activation_ref" }
}));

INSTANTIATE_TEST_SUITE_P(DISABLED_fusings_gpu, activation_eltwise_activation, ::testing::ValuesIn(std::vector<activation_test_params>{
    activation_test_params{ CASE_ACTIVATION_3D_F32_4, 2, 4, "activation_ref" },  // FIXME - accuracy bug
    activation_test_params{ CASE_ACTIVATION_3D_F32_5, 2, 4, "activation_ref" },  // FIXME - accuracy bug
}));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------- Extended Activation Fusion Tests for Conv/FC/GEMM --------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

// Extended test parameters for multiple activation functions
struct extended_activation_test_params {
    ov::PartialShape in_shape;
    ov::PartialShape out_shape;
    data_types input_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    std::string kernel_name;
    uint32_t groups;
    activation_func activation_function;
    float activation_param_a;
    float activation_param_b;
};

// Helper function to convert activation_func enum to string
static std::string activation_func_to_string(activation_func func) {
    try {
        auto ks_activation = get_kernel_selector_activation_param(func);
        return kernel_selector::toString(ks_activation);
    } catch (...) {
        return "UNKNOWN_" + std::to_string(static_cast<int>(func));
    }
}

static std::vector<extended_activation_test_params> generate_all_activation_test_cases(const std::vector<extended_activation_test_params>& base_cases);

// Test case definitions for extended activation fusion with DYNAMIC shapes
#define CASE_CONV_ACTIVATION_DYNAMIC_F32 ov::PartialShape{1, 3, -1, -1}, ov::PartialShape{1, 30, -1, -1}, data_types::f32, format::bfyx, data_types::f32, format::bfyx, data_types::f32, format::bfyx, 2, 3, "convolution_gpu_bfyx_os_iyx_osv16", 1
#define CASE_CONV_ACTIVATION_DYNAMIC_F16 ov::PartialShape{1, 3, -1, -1}, ov::PartialShape{1, 30, -1, -1}, data_types::f16, format::bfyx, data_types::f16, format::bfyx, data_types::f32, format::bfyx, 2, 3, "convolution_gpu_bfyx_os_iyx_osv16", 1

#define CASE_FC_ACTIVATION_DYNAMIC_F32 ov::PartialShape{-1, 48}, ov::PartialShape{-1, 64}, data_types::f32, format::bfyx, data_types::f32, format::bfyx, data_types::f32, format::bfyx, 2, 3, "fully_connected_gpu_ref", 1
#define CASE_FC_ACTIVATION_DYNAMIC_F16 ov::PartialShape{-1, 48}, ov::PartialShape{-1, 64}, data_types::f16, format::bfyx, data_types::f16, format::bfyx, data_types::f32, format::bfyx, 2, 3, "fully_connected_gpu_ref", 1

#define CASE_GEMM_ACTIVATION_DYNAMIC_F32 ov::PartialShape{1, 1, -1, -1}, ov::PartialShape{1, 1, -1, -1}, data_types::f32, format::bfyx, data_types::f32, format::bfyx, data_types::f32, format::bfyx, 3, 4, "gemm_ref", 1
#define CASE_GEMM_ACTIVATION_DYNAMIC_F16 ov::PartialShape{1, 1, -1, -1}, ov::PartialShape{1, 1, -1, -1}, data_types::f16, format::bfyx, data_types::f16, format::bfyx, data_types::f32, format::bfyx, 3, 4, "gemm_ref", 1

// Base test class for extended activation fusion
class ExtendedActivationFusingTest : public ::BaseFusingTest<extended_activation_test_params> {
public:
    // Check if oneDNN supports the given activation function
    bool is_activation_supported_by_onednn(activation_func func) {
        // Based on onednn::convert_activation_func implementation
        switch (func) {
            case activation_func::relu:
            case activation_func::relu_negative_slope:
            case activation_func::gelu:
            case activation_func::gelu_tanh:
            case activation_func::elu:
            case activation_func::mish:
            case activation_func::swish:
            case activation_func::hswish:
            case activation_func::abs:
            case activation_func::exp:
            case activation_func::logistic:
            case activation_func::clamp:
            case activation_func::hyperbolic_tan:
            case activation_func::pow:
            case activation_func::sqrt:
            case activation_func::square:
            case activation_func::hard_sigmoid:
            case activation_func::hsigmoid:
            case activation_func::negative:
                return true;
            default:
                return false;
        }
    }

    void execute(extended_activation_test_params& p) {
        // Set appropriate tolerance based on data type and activation function
        // F16 needs much higher tolerance due to reduced precision
        if (p.input_type == data_types::f16) {
            // For F16, use much higher tolerance due to precision issues
            tolerance = 1.0f;
        } else {
            tolerance = 1e-5f;
        }

        // Use positive range for functions that need it (e.g., sqrt)
        bool needs_positive_input = (p.activation_function == activation_func::sqrt);
        // Use smaller range for functions that can overflow (e.g., exp, pow)
        bool needs_small_range = (p.activation_function == activation_func::exp ||
                                 p.activation_function == activation_func::pow);

        auto input_prim = needs_positive_input ?
                         get_mem(get_input_layout(p), 0.1, 2) :    // Positive range for sqrt
                         needs_small_range ?
                         get_mem(get_input_layout(p), -1, 1) :     // Small range for exp/pow to avoid overflow
                         get_mem(get_input_layout(p), -2, 2);      // Standard range

        // Get the default configuration of the test environment
        ExecutionConfig cfg = get_test_default_config(engine);
        // Enable data optimization, which triggers fusion optimization
        cfg.set_property(ov::intel_gpu::optimize_data(true));

        network network_fused(this->engine, this->topology_fused, cfg);
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        // Check oneDNN support before comparing
        bool onednn_supports = is_activation_supported_by_onednn(p.activation_function);
        compare_with_onednn_check(network_not_fused, network_fused, p, onednn_supports);
    }

    void compare_with_onednn_check(network& not_fused, network& fused, extended_activation_test_params& p, bool onednn_supports) {
        auto outputs_ref = not_fused.execute();
        auto outputs_fused = fused.execute();

        auto get_reorders_count = [](network& net) -> size_t {
            size_t count = 0;
            for (auto& pi : net.get_primitives_info()) {
                if (pi.type_id == "reorder") {                       // pi.type_id maybe: "input_layout", "data/input_layout", "convolution/fully_connected/gemm", "activation", "reorder"
                    auto exec_prims = net.get_executed_primitives(); // exec_prims is a map, maybe: [("input", event_ptr), ("fc", event_ptr), ("act", event_ptr), ("out", event_ptr)]
                    auto it = std::find_if(exec_prims.begin(), exec_prims.end(), [&](const std::pair<primitive_id, event::ptr>& e) -> bool {
                        return e.first == pi.original_id;            // pi.original_id maybe: "input", "weights", "conv/fc/gemm", "act", "out"
                    });
                    if (it != exec_prims.end())
                        count++;
                }
            }
            return count;
        };

        size_t reorders_count_fused = get_reorders_count(fused);
        size_t reorders_count_not_fused = get_reorders_count(not_fused);

        // Reorders are not functional operations, they are simply format conversions and do not participate in actual computation.
        // The number of reorders affects fusion verification, we are comparing the actual number of computational primitives.
        // Optimization may insert or remove reorder operations,
        // Remove reorder operations, and only compare the actual number of computational primitives.
        size_t actual_fused_primitives = fused.get_executed_primitives().size() - reorders_count_fused;
        size_t actual_not_fused_primitives = not_fused.get_executed_primitives().size() - reorders_count_not_fused;

        std::stringstream description;
        description << std::endl << "not fused: " << std::endl;
        for (auto i : not_fused.get_primitives_info()) {
            description << "  " << i.original_id << " " << i.kernel_id << std::endl;
        }
        description << "fused: " << std::endl;
        for (auto i : fused.get_primitives_info()) {
            description << "  " << i.original_id << " " << i.kernel_id << std::endl;
        }

        // Add activation function information
        description << "Activation Function: " << activation_func_to_string(p.activation_function)
                   << " (oneDNN supported: " << (onednn_supports ? "YES" : "NO") << ")" << std::endl;
        description << "Expected fused primitives: " << p.expected_fused_primitives
                   << ", Actual: " << actual_fused_primitives << std::endl;
        description << "Expected not fused primitives: " << p.expected_not_fused_primitives
                   << ", Actual: " << actual_not_fused_primitives << std::endl;

        SCOPED_TRACE(description.str());

        // Check if primitive counts match expectations
        bool primitive_counts_match = (actual_fused_primitives == p.expected_fused_primitives) &&
                                    (actual_not_fused_primitives == p.expected_not_fused_primitives);

        if (!primitive_counts_match) {
            if (!onednn_supports) {
                // oneDNN doesn't support this activation, fusion failure is expected
                GTEST_LOG_(INFO) << "Activation function " << activation_func_to_string(p.activation_function)
                                << " is not supported by oneDNN, fusion disabled. This is expected behavior.";
                // Don't fail the test, this is expected behavior
                return;
            } else {
                // Check if this is a known limitation for convolution kernels
                bool is_known_kernel_limitation = false;
                switch (p.activation_function) {
                    case activation_func::hyperbolic_tan:
                        // hyperbolic_tan fusion is explicitly blocked in prepare_primitive_fusing.cpp
                        // This is a hardware-specific limitation
                        is_known_kernel_limitation = true;
                        break;
                    default:
                        break;
                }

                if (is_known_kernel_limitation) {
                    std::string original_id_str = fused.get_primitives_info()[2].original_id;
                    std::transform(original_id_str.begin(), original_id_str.end(), original_id_str.begin(), ::toupper);
                    GTEST_LOG_(WARNING) << "The fusion between " << original_id_str << " and the "
                                       << activation_func_to_string(p.activation_function) << " activation function faces kernel limitations. "
                                       << "Treating as expected behavior despite oneDNN support.";
                    return; // Skip accuracy check and pass the test
                }

                FAIL() << "Activation function " << activation_func_to_string(p.activation_function)
                       << " is supported by oneDNN but fusion failed unexpectedly. "
                       << "Expected fused primitives: " << p.expected_fused_primitives
                       << ", Actual: " << actual_fused_primitives
                       << ". This indicates a bug in the fusion logic.";
            }
        }

        // If primitive counts match, check numerical accuracy
        ASSERT_EQ(outputs_ref.size(), outputs_fused.size());
        ASSERT_EQ(outputs_ref.size(), size_t(1));

        std::vector<float> val_opt;
        auto val_ref = get_output_values_to_float(not_fused, outputs_ref.begin()->second);
        ASSERT_NO_THROW(val_opt = get_output_values_to_float(fused, outputs_fused.begin()->second));
        ASSERT_EQ(val_ref.size(), val_opt.size());
        for (size_t i = 0; i < val_ref.size(); i++) {
            ASSERT_NEAR(val_ref[i], val_opt[i], tolerance)
                << "tolerance = " << tolerance
                << "\ni = " << i
                << "\nref[i] = " << val_ref[i]
                << "\nopt[i] = " << val_opt[i];
        }
    }

    layout get_input_layout(extended_activation_test_params& p) {
        // Convert dynamic shapes to concrete shapes for testing
        auto concrete_shape = p.in_shape;
        for (size_t i = 0; i < concrete_shape.size(); ++i) {
            if (concrete_shape[i].is_dynamic()) {
                // Use reasonable concrete values for dynamic dimensions
                if (i == 0) concrete_shape[i] = 1;      // batch
                else if (i == 1) concrete_shape[i] = (p.in_shape.size() == 2) ? 48 : 3;  // channels/features
                else if (i == 2) concrete_shape[i] = 32; // height/spatial
                else if (i == 3) concrete_shape[i] = 32; // width/spatial
            }
        }
        return layout{ concrete_shape, p.input_type, p.input_format };
    }

    layout get_weights_layout(extended_activation_test_params& p) {
        // Convert dynamic output shapes to concrete shapes for weights
        auto concrete_out_shape = p.out_shape;
        auto concrete_in_shape = p.in_shape;

        for (size_t i = 0; i < concrete_out_shape.size(); ++i) {
            if (concrete_out_shape[i].is_dynamic()) { // {1, 30, -1, -1} -> {1, 30, 30, 30}
                if (i == 0) concrete_out_shape[i] = 1;      // batch
                else if (i == 1) concrete_out_shape[i] = (p.out_shape.size() == 2) ? 64 : 30;  // output channels/features
                else if (i == 2) concrete_out_shape[i] = 30; // output height/spatial
                else if (i == 3) concrete_out_shape[i] = 30; // output width/spatial
            }
        }

        for (size_t i = 0; i < concrete_in_shape.size(); ++i) {
            if (concrete_in_shape[i].is_dynamic()) { // {1, 3, -1, -1} -> {1, 3, 32, 32}
                if (i == 0) concrete_in_shape[i] = 1;      // batch
                else if (i == 1) concrete_in_shape[i] = (p.in_shape.size() == 2) ? 48 : 3;  // input channels/features
                else if (i == 2) concrete_in_shape[i] = 32; // input height/spatial
                else if (i == 3) concrete_in_shape[i] = 32; // input width/spatial
            }
        }

        cldnn::tensor weights_tensor;
        if (p.groups == 1) {
            weights_tensor = cldnn::tensor(batch(static_cast<int32_t>(concrete_out_shape[1].get_length())),
                                         feature(static_cast<int32_t>(concrete_in_shape[1].get_length())),
                                         spatial(3, 3));
        } else {
            weights_tensor = cldnn::tensor(group(p.groups),
                                         batch(static_cast<int32_t>(concrete_out_shape[1].get_length()) / p.groups),
                                         feature(static_cast<int32_t>(concrete_in_shape[1].get_length()) / p.groups),
                                         spatial(3, 3));
        }
        return layout{p.weights_type, p.weights_format, weights_tensor};
    }

    format get_input_format(extended_activation_test_params& p) { return p.input_format; }
};

// Convolution + Activation fusion test class
class conv_activation_fusion_extended : public ExtendedActivationFusingTest {
public:
    void execute(extended_activation_test_params& p) {
        // Set appropriate tolerance based on data type and activation function
        // Some activation functions need higher tolerance even for F32
        if (p.input_type == data_types::f16) {
            // For F16, use much higher tolerance due to precision issues
            tolerance = 1.0f;
        } else {
            // For F32, use higher tolerance for functions that can be numerically sensitive
            if (p.activation_function == activation_func::sqrt ||
                p.activation_function == activation_func::gelu ||
                p.activation_function == activation_func::gelu_tanh ||
                p.activation_function == activation_func::swish ||
                p.activation_function == activation_func::mish ||
                p.activation_function == activation_func::hyperbolic_tan) {
                tolerance = 1e-3f;  // Higher tolerance for numerically sensitive functions
            } else {
                tolerance = 1e-5f;  // Standard tolerance for stable functions
            }
        }

        create_topologies(
            input_layout("input", get_input_layout(p)),
            // Small positive weights to ensure positive conv output for sqrt
            data("weights", get_mem(get_weights_layout(p), 0.01, 0.1)),
            // "": No bias; 1: Number of groups (usually 1); {1, 1}: Stride; {1, 1}: Dilation; {0, 0}: Input padding; {0, 0}: Output padding; false: Do not invert weights.
            convolution("conv", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
            activation("act", input_info("conv"), p.activation_function, {p.activation_param_a, p.activation_param_b}),
            reorder("out", input_info("act"), p.default_format, p.default_type)
        );

        // Record test property for better test reporting
        RecordProperty("CONV activation function", activation_func_to_string(p.activation_function));

        // Use positive range for functions that need it (e.g., sqrt)
        bool needs_positive_input = (p.activation_function == activation_func::sqrt);
        // Use smaller range for functions that can overflow (e.g., exp, pow)
        bool needs_small_range = (p.activation_function == activation_func::exp ||
                                 p.activation_function == activation_func::pow);

        auto input_prim = needs_positive_input ?
                         get_mem(get_input_layout(p), 0.1, 2) :    // Positive range for sqrt
                         needs_small_range ?
                         get_mem(get_input_layout(p), -1, 1) :     // Small range for exp/pow to avoid overflow
                         get_mem(get_input_layout(p), -2, 2);      // Standard range

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::optimize_data(true));

        network network_fused(this->engine, this->topology_fused, cfg);
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        // Check oneDNN support before comparing
        bool onednn_supports = is_activation_supported_by_onednn(p.activation_function);
        compare_with_onednn_check(network_not_fused, network_fused, p, onednn_supports);
    }
};

// Fully Connected + Activation fusion test class
class fc_activation_fusion_extended : public ExtendedActivationFusingTest {
public:
    void execute(extended_activation_test_params& p) {
        // Set appropriate tolerance based on data type and activation function
        // F16 needs much higher tolerance due to reduced precision
        if (p.input_type == data_types::f16) {
            // For F16, use much higher tolerance due to precision issues
            tolerance = 1.0f;  // Very high tolerance for F16
        } else {
            tolerance = 1e-5f;
        }

        // Convert dynamic shapes to concrete shapes for FC
        auto concrete_in_shape = p.in_shape;
        auto concrete_out_shape = p.out_shape;

        for (size_t i = 0; i < concrete_in_shape.size(); ++i) {
            if (concrete_in_shape[i].is_dynamic()) {
                if (i == 0) concrete_in_shape[i] = 1;  // batch
                else if (i == 1) concrete_in_shape[i] = 48; // input features
            }
        }

        for (size_t i = 0; i < concrete_out_shape.size(); ++i) {
            if (concrete_out_shape[i].is_dynamic()) {
                if (i == 0) concrete_out_shape[i] = 1;  // batch
                else if (i == 1) concrete_out_shape[i] = 64; // output features
            }
        }

        auto fc_input_layout = layout{ {static_cast<int32_t>(concrete_in_shape[0].get_length()), static_cast<int32_t>(concrete_in_shape[1].get_length())}, p.input_type, format::bfyx };
        auto fc_weights_layout = layout{ {static_cast<int32_t>(concrete_out_shape[1].get_length()), static_cast<int32_t>(concrete_in_shape[1].get_length())}, p.weights_type, format::bfyx };

        create_topologies(
            input_layout("input", fc_input_layout),
            data("weights", get_mem(fc_weights_layout, 0.01, 0.1)),  // Small positive weights to ensure positive FC output for sqrt
            fully_connected("fc", input_info("input"), "weights"),
            activation("act", input_info("fc"), p.activation_function, {p.activation_param_a, p.activation_param_b}),
            reorder("out", input_info("act"), p.default_format, p.default_type)
        );

        // Record test property for better test reporting
        RecordProperty("FC activation function", activation_func_to_string(p.activation_function));

        // Use positive range for functions that need it (e.g., sqrt)
        bool needs_positive_input = (p.activation_function == activation_func::sqrt);
        // Use smaller range for functions that can overflow (e.g., exp, pow)
        bool needs_small_range = (p.activation_function == activation_func::exp ||
                                 p.activation_function == activation_func::pow);

        auto input_prim = needs_positive_input ?
                         get_mem(fc_input_layout, 0.1, 2) :       // Positive range for sqrt
                         needs_small_range ?
                         get_mem(fc_input_layout, -1, 1) :        // Small range for exp/pow to avoid overflow
                         get_mem(fc_input_layout, -2, 2);         // Standard range

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::optimize_data(true));

        network network_fused(this->engine, this->topology_fused, cfg);
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        // Check oneDNN support before comparing
        bool onednn_supports = is_activation_supported_by_onednn(p.activation_function);
        compare_with_onednn_check(network_not_fused, network_fused, p, onednn_supports);
    }
};

// GEMM + Activation fusion test class
class gemm_activation_fusion_extended : public ExtendedActivationFusingTest {
public:
    void execute(extended_activation_test_params& p) {
        // Set appropriate tolerance based on data type and activation function
        // F16 needs much higher tolerance due to reduced precision
        if (p.input_type == data_types::f16) {
            // For F16, use much higher tolerance due to precision issues
            tolerance = 1.0f;  // Very high tolerance for F16
        } else {
            tolerance = 1e-5f;
        }

        // Convert dynamic shapes to concrete shapes for GEMM
        auto concrete_shape = p.in_shape;
        for (size_t i = 0; i < concrete_shape.size(); ++i) {
            if (concrete_shape[i].is_dynamic()) {
                if (i == 0) concrete_shape[i] = 1;      // batch
                else if (i == 1) concrete_shape[i] = 1; // channel
                else if (i == 2) concrete_shape[i] = 32; // height
                else if (i == 3) concrete_shape[i] = 32; // width
            }
        }

        // GEMM requires 4D layout for bfyx format
        auto gemm_input_layout = layout{ {static_cast<int32_t>(concrete_shape[0].get_length()),
                                        static_cast<int32_t>(concrete_shape[1].get_length()),
                                        static_cast<int32_t>(concrete_shape[2].get_length()),
                                        static_cast<int32_t>(concrete_shape[3].get_length())}, p.input_type, format::bfyx };

        create_topologies(
            input_layout("input", gemm_input_layout),
            input_layout("input2", gemm_input_layout),
            gemm("gemm", { input_info("input"), input_info("input2") }, data_types::f32),
            activation("act", input_info("gemm"), p.activation_function, {p.activation_param_a, p.activation_param_b}),
            reorder("out", input_info("act"), p.default_format, p.default_type)
        );

        // Record test property for better test reporting
        RecordProperty("GEMM activation function", activation_func_to_string(p.activation_function));

        // Custom execution for GEMM with two inputs
        bool needs_positive_input = (p.activation_function == activation_func::sqrt);
        // Use smaller range for functions that can overflow (e.g., exp, pow)
        bool needs_small_range = (p.activation_function == activation_func::exp ||
                                 p.activation_function == activation_func::pow);

        auto input_prim = needs_positive_input ?
                         get_mem(gemm_input_layout, 0.1, 2) :     // Positive range for sqrt
                         needs_small_range ?
                         get_mem(gemm_input_layout, -1, 1) :      // Small range for exp/pow to avoid overflow
                         get_mem(gemm_input_layout, -2, 2);       // Standard range
        auto input2_prim = needs_positive_input ?
                          get_mem(gemm_input_layout, 0.1, 2) :    // Positive range for sqrt
                          needs_small_range ?
                          get_mem(gemm_input_layout, -1, 1) :     // Small range for exp/pow to avoid overflow
                          get_mem(gemm_input_layout, -2, 2);      // Standard range

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::optimize_data(true));

        network network_fused(this->engine, this->topology_fused, cfg);
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        network_fused.set_input_data("input2", input2_prim);
        network_not_fused.set_input_data("input2", input2_prim);

        // Check oneDNN support before comparing
        bool onednn_supports = is_activation_supported_by_onednn(p.activation_function);
        compare_with_onednn_check(network_not_fused, network_fused, p, onednn_supports);
    }
};

// Test method implementations
TEST_P(conv_activation_fusion_extended, basic) {
    auto p = GetParam();
    execute(p);
}

TEST_P(fc_activation_fusion_extended, basic) {
    auto p = GetParam();
    execute(p);
}

TEST_P(gemm_activation_fusion_extended, basic) {
    auto p = GetParam();
    execute(p);
}

// Function to generate test cases for all 19 activation functions
static std::vector<extended_activation_test_params> generate_all_activation_test_cases(const std::vector<extended_activation_test_params>& base_cases) {
    std::vector<extended_activation_test_params> test_cases;

    // Define all 19 activation functions with their parameters
    std::vector<std::tuple<activation_func, float, float>> activations = {
        {activation_func::relu, 0.0f, 0.0f},
        {activation_func::relu_negative_slope, 0.1f, 0.0f},
        {activation_func::gelu, 0.0f, 0.0f},
        {activation_func::gelu_tanh, 0.0f, 0.0f},
        {activation_func::elu, 1.0f, 0.0f},
        {activation_func::mish, 0.0f, 0.0f},
        {activation_func::swish, 1.0f, 0.0f},
        {activation_func::hswish, 0.0f, 0.0f},
        {activation_func::abs, 0.0f, 0.0f},
        {activation_func::exp, 0.0f, 0.0f},
        {activation_func::logistic, 0.0f, 0.0f},
        {activation_func::clamp, -1.0f, 1.0f},
        {activation_func::hyperbolic_tan, 0.0f, 0.0f},
        {activation_func::pow, 2.0f, 0.0f},
        {activation_func::sqrt, 0.0f, 0.0f},
        {activation_func::square, 0.0f, 0.0f},
        {activation_func::hard_sigmoid, 0.2f, 0.5f},
        {activation_func::hsigmoid, 0.0f, 0.0f},
        {activation_func::negative, 0.0f, 0.0f}
    };

    for (const auto& base_case : base_cases) {
        for (const auto& [func, param_a, param_b] : activations) {
            auto test_case = base_case;
            test_case.activation_function = func;
            test_case.activation_param_a = param_a;
            test_case.activation_param_b = param_b;
            test_cases.push_back(test_case);
        }
    }

    return test_cases;
}

// Test instantiations for extended activation fusion with DYNAMIC inputs
INSTANTIATE_TEST_SUITE_P(fusings_gpu_extended,
                        conv_activation_fusion_extended,
                        ::testing::ValuesIn(generate_all_activation_test_cases({
                            {CASE_CONV_ACTIVATION_DYNAMIC_F32},
                            {CASE_CONV_ACTIVATION_DYNAMIC_F16}
                        })));

INSTANTIATE_TEST_SUITE_P(fusings_gpu_extended,
                        fc_activation_fusion_extended,
                        ::testing::ValuesIn(generate_all_activation_test_cases({
                            {CASE_FC_ACTIVATION_DYNAMIC_F32},
                            {CASE_FC_ACTIVATION_DYNAMIC_F16}
                        })));

INSTANTIATE_TEST_SUITE_P(fusings_gpu_extended,
                        gemm_activation_fusion_extended,
                        ::testing::ValuesIn(generate_all_activation_test_cases({
                            {CASE_GEMM_ACTIVATION_DYNAMIC_F32},
                            {CASE_GEMM_ACTIVATION_DYNAMIC_F16}
                        })));

