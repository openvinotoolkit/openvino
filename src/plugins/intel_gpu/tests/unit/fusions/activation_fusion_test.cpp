// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>

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
