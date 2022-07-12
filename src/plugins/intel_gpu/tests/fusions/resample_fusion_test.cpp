// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/resample.hpp>
#include <intel_gpu/primitives/reshape.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct resample_test_params {
    tensor in_shape;
    tensor out_shape;
    data_types data_type;
    format input_format;
    resample_type type;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class ResamplePrimitiveFusingTest : public ::BaseFusingTest<resample_test_params> {
public:

    void execute(resample_test_params& p, std::map<std::string, std::vector<std::string>> expected_fused_primitives_ids = {}) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        check_fusions_correctness(network_fused, expected_fused_primitives_ids);
    }

    layout get_input_layout(resample_test_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape, padding{} };
    }

    layout get_per_channel_layout(resample_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.out_shape.feature[0], 1, 1 } };
    }
};

}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- Resample cases --------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_RESAMPLE_FP32_1 { 1, 15, 4, 5 }, { 1, 15, 2, 3 }, data_types::f32, format::bfyx, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_2 { 1, 15, 4, 5 }, { 1, 15, 2, 3 }, data_types::f32, format::bfyx, resample_type::bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_3 { 1, 15, 4, 5 }, { 1, 15, 2, 3 }, data_types::f32, format::bfyx, resample_type::caffe_bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_4 { 1, 16, 4, 5 }, { 1, 16, 7, 8 }, data_types::f32, format::bfyx, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_5 { 1, 16, 4, 5 }, { 1, 16, 7, 8 }, data_types::f32, format::bfyx, resample_type::bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_6 { 1, 16, 4, 5 }, { 1, 16, 7, 8 }, data_types::f32, format::bfyx, resample_type::caffe_bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_7 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, data_types::f32, format::bfzyx, resample_type::nearest, data_types::f32, format::bfzyx
#define CASE_RESAMPLE_FP32_8 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, data_types::f32, format::bfzyx, resample_type::caffe_bilinear, data_types::f32, format::bfzyx
#define CASE_RESAMPLE_FP32_9 { 1, 16, 4, 5 }, { 1, 16, 7, 8 }, data_types::f32, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_10 { 1, 16, 4, 5 }, { 1, 16, 7, 8 }, data_types::f32, format::b_fs_yx_fsv16, resample_type::caffe_bilinear, data_types::f32, format::bfyx

#define CASE_RESAMPLE_FP16_1 { 1, 15, 4, 5 }, { 1, 15, 2, 3 }, data_types::f16, format::bfyx, resample_type::nearest, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_2 { 1, 15, 4, 5 }, { 1, 15, 2, 3 }, data_types::f16, format::bfyx, resample_type::bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_3 { 1, 15, 4, 5 }, { 1, 15, 2, 3 }, data_types::f16, format::bfyx, resample_type::caffe_bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_4 { 1, 16, 4, 5 }, { 1, 16, 7, 8 }, data_types::f16, format::bfyx, resample_type::nearest, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_5 { 1, 16, 4, 5 }, { 1, 16, 7, 8 }, data_types::f16, format::bfyx, resample_type::bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_6 { 1, 16, 4, 5 }, { 1, 16, 7, 8 }, data_types::f16, format::bfyx, resample_type::caffe_bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_7 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, data_types::f16, format::bfzyx, resample_type::nearest, data_types::f16, format::bfzyx
#define CASE_RESAMPLE_FP16_8 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, data_types::f16, format::bfzyx, resample_type::caffe_bilinear, data_types::f16, format::bfzyx
#define CASE_RESAMPLE_FP16_9 { 1, 16, 4, 5 }, { 1, 16, 7, 8 }, data_types::f16, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_10 { 2, 32, 4, 5 }, { 2, 32, 7, 8 }, data_types::f16, format::fs_b_yx_fsv32, resample_type::bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_11 { 1, 16, 4, 5 }, { 1, 16, 7, 8 }, data_types::f16, format::b_fs_yx_fsv16, resample_type::caffe_bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_12 { 2, 32, 4, 5 }, { 2, 32, 7, 8 }, data_types::f16, format::fs_b_yx_fsv32, resample_type::caffe_bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_13 { 1, 16, 4, 5 }, { 1, 16, 7, 8 }, data_types::f16, format::b_fs_yx_fsv16, resample_type::caffe_bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_14 { 1, 32, 4, 5 }, { 1, 32, 2, 3 }, data_types::f16, format::fs_b_yx_fsv32, resample_type::caffe_bilinear, data_types::f16, format::bfyx

#define CASE_RESAMPLE_I8_1 { 1, 16, 4, 5 }, { 1, 16, 2, 3 }, data_types::i8, format::b_fs_yx_fsv16, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_I8_2 { 2, 32, 4, 5 }, { 2, 32, 2, 3 }, data_types::i8, format::b_fs_yx_fsv16, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_I8_3 { 1, 16, 4, 5 }, { 1, 16, 2, 3 }, data_types::i8, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_I8_4 { 2, 32, 4, 5 }, { 2, 32, 2, 3 }, data_types::i8, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f32, format::bfyx

#define CASE_RESAMPLE_U8_1 { 1, 16, 4, 5 }, { 1, 16, 2, 3 }, data_types::u8, format::b_fs_yx_fsv16, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_U8_2 { 2, 32, 4, 5 }, { 2, 32, 2, 3 }, data_types::u8, format::b_fs_yx_fsv16, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_U8_3 { 1, 16, 4, 5 }, { 1, 16, 2, 3 }, data_types::u8, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_U8_4 { 2, 32, 4, 5 }, { 2, 32, 2, 3 }, data_types::u8, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f32, format::bfyx

class resample_quantize : public ResamplePrimitiveFusingTest {};
TEST_P(resample_quantize, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        resample("resample_prim", "input", p.out_shape, p.in_shape.feature[0], p.type),
        quantize("quantize", "resample_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

#define RESAMPLE_QUANTIZE_CNT 2, 3
INSTANTIATE_TEST_SUITE_P(fusings_gpu, resample_quantize, ::testing::ValuesIn(std::vector<resample_test_params>{
    resample_test_params{ CASE_RESAMPLE_FP32_1, RESAMPLE_QUANTIZE_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_2, RESAMPLE_QUANTIZE_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_3, RESAMPLE_QUANTIZE_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_4, RESAMPLE_QUANTIZE_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_5, RESAMPLE_QUANTIZE_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_6, RESAMPLE_QUANTIZE_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_7, RESAMPLE_QUANTIZE_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_8, RESAMPLE_QUANTIZE_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_9, RESAMPLE_QUANTIZE_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_10, RESAMPLE_QUANTIZE_CNT },

    // FQ can't be fused to FP16 primitive for now
    // resample_test_params{ CASE_RESAMPLE_FP16_1, RESAMPLE_QUANTIZE_CNT },
    // resample_test_params{ CASE_RESAMPLE_FP16_2, RESAMPLE_QUANTIZE_CNT },
    // resample_test_params{ CASE_RESAMPLE_FP16_3, RESAMPLE_QUANTIZE_CNT },
    // resample_test_params{ CASE_RESAMPLE_FP16_4, RESAMPLE_QUANTIZE_CNT },
    // resample_test_params{ CASE_RESAMPLE_FP16_5, RESAMPLE_QUANTIZE_CNT },
    // resample_test_params{ CASE_RESAMPLE_FP16_6, RESAMPLE_QUANTIZE_CNT },
    // resample_test_params{ CASE_RESAMPLE_FP16_7, RESAMPLE_QUANTIZE_CNT },
    // resample_test_params{ CASE_RESAMPLE_FP16_8, RESAMPLE_QUANTIZE_CNT },
    // resample_test_params{ CASE_RESAMPLE_FP16_9, RESAMPLE_QUANTIZE_CNT },
}));

class resample_scale_activation_eltwise : public ResamplePrimitiveFusingTest {};
TEST_P(resample_scale_activation_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
        data("eltwise_data", get_mem(get_output_layout(p), -10, 10)),
        resample("resample_prim", "input", p.out_shape, p.in_shape.feature[0], p.type),
        eltwise("scale", { "resample_prim", "scale_data" }, eltwise_mode::prod, data_types::f16),
        activation("activation", "scale", activation_func::abs),
        eltwise("eltwise", { "activation", "eltwise_data" }, eltwise_mode::sum),
        reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

#define RESAMPLE_SCALE_ACTIVATION_ELTWISE 2, 5
INSTANTIATE_TEST_SUITE_P(fusings_gpu, resample_scale_activation_eltwise, ::testing::ValuesIn(std::vector<resample_test_params>{
    resample_test_params{ CASE_RESAMPLE_FP32_1, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP32_2, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP32_3, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP32_4, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP32_5, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP32_6, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP32_7, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP32_8, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP32_9, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP32_10, RESAMPLE_SCALE_ACTIVATION_ELTWISE },

    resample_test_params{ CASE_RESAMPLE_FP16_1, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_2, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_3, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_4, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_5, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_6, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_7, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_8, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_9, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_10, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_11, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_12, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_13, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_FP16_14, RESAMPLE_SCALE_ACTIVATION_ELTWISE },

    resample_test_params{ CASE_RESAMPLE_I8_1, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_I8_2, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_I8_3, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_I8_4, RESAMPLE_SCALE_ACTIVATION_ELTWISE },

    resample_test_params{ CASE_RESAMPLE_U8_1, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_U8_2, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_U8_3, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
    resample_test_params{ CASE_RESAMPLE_U8_4, RESAMPLE_SCALE_ACTIVATION_ELTWISE },
}));

class resample_quantize_concat : public ResamplePrimitiveFusingTest {};
TEST_P(resample_quantize_concat, along_f) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        resample("resample1", "input", p.out_shape, p.in_shape.feature[0], p.type),
        data("in_lo_1", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi_1", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo_1", get_mem(get_single_element_layout(p), -128)),
        data("out_hi_1", get_mem(get_single_element_layout(p), 127)),
        quantize("quant1", "resample1", "in_lo_1", "in_hi_1", "out_lo_1", "out_hi_1", 256, data_types::i8),
        resample("resample2", "input", p.out_shape, p.in_shape.feature[0], p.type),
        data("in_lo_2", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi_2", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo_2", get_mem(get_single_element_layout(p), -127)),
        data("out_hi_2", get_mem(get_single_element_layout(p), 127)),
        quantize("quant2", "resample2", "in_lo_2", "in_hi_2", "out_lo_2", "out_hi_2", 255, data_types::i8),
        concatenation("concat", { "quant1", "quant2" }, 1),
        reorder("reorder_bfyx", "concat", cldnn::format::bfyx, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

#define RESAMPLE_QUANTIZE_CONCAT_CNT 3, 6
INSTANTIATE_TEST_SUITE_P(fusings_gpu, resample_quantize_concat, ::testing::ValuesIn(std::vector<resample_test_params>{
    resample_test_params{ CASE_RESAMPLE_FP32_1, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_2, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_3, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_4, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_5, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_6, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_7, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_8, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_9, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_10, RESAMPLE_QUANTIZE_CONCAT_CNT },

    resample_test_params{ CASE_RESAMPLE_FP16_1, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_2, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_3, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_4, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_5, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_6, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_7, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_8, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_9, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_10, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_11, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_12, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_13, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_14, RESAMPLE_QUANTIZE_CONCAT_CNT },

    resample_test_params{ CASE_RESAMPLE_I8_3, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_I8_4, RESAMPLE_QUANTIZE_CONCAT_CNT },

    resample_test_params{ CASE_RESAMPLE_U8_3, RESAMPLE_QUANTIZE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_U8_4, RESAMPLE_QUANTIZE_CONCAT_CNT },
}));

class resample_eltwise_concat : public ResamplePrimitiveFusingTest {};
TEST_P(resample_eltwise_concat, along_f) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        resample("resample1", "input", p.out_shape, p.in_shape.feature[0], p.type),
        data("eltwise1_scale", get_mem(get_per_channel_layout(p), -10, 10)),
        data("eltwise1_shift", get_mem(get_per_channel_layout(p), -10, 10)),
        eltwise("eltwise1_bias", { "resample1", "eltwise1_shift" }, eltwise_mode::sum),
        eltwise("eltwise1", { "eltwise1_bias", "eltwise1_scale" }, eltwise_mode::prod),
        resample("resample2", "input", p.out_shape, p.in_shape.feature[0], p.type),
        data("eltwise2_scale", get_mem(get_per_channel_layout(p), -10, 10)),
        data("eltwise2_shift", get_mem(get_per_channel_layout(p), -10, 10)),
        eltwise("eltwise2_bias", { "resample2", "eltwise2_shift" }, eltwise_mode::sum),
        eltwise("eltwise2", { "eltwise2_bias", "eltwise2_scale" }, eltwise_mode::prod),
        concatenation("concat", { "eltwise1", "eltwise2" }, 1),
        reorder("reorder_bfyx", "concat", cldnn::format::bfyx, p.default_type)
    );

    tolerance = 1e-5f;
    execute(p);
}

#define RESAMPLE_ELTWISE_CONCAT_CNT 3, 8
INSTANTIATE_TEST_SUITE_P(fusings_gpu, resample_eltwise_concat, ::testing::ValuesIn(std::vector<resample_test_params>{
    resample_test_params{ CASE_RESAMPLE_FP32_1, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_2, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_3, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_4, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_5, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_6, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_7, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_8, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_9, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_10, RESAMPLE_ELTWISE_CONCAT_CNT },

    resample_test_params{ CASE_RESAMPLE_FP16_1, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_2, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_3, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_4, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_5, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_6, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_7, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_8, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_9, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_10, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_11, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_12, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_13, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_14, RESAMPLE_ELTWISE_CONCAT_CNT },

    resample_test_params{ CASE_RESAMPLE_I8_1, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_I8_2, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_I8_3, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_I8_4, RESAMPLE_ELTWISE_CONCAT_CNT },

    resample_test_params{ CASE_RESAMPLE_U8_1, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_U8_2, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_U8_3, RESAMPLE_ELTWISE_CONCAT_CNT },
    resample_test_params{ CASE_RESAMPLE_U8_4, RESAMPLE_ELTWISE_CONCAT_CNT },
}));

class resample_eltwise_fusing_through : public ResamplePrimitiveFusingTest {};
TEST_P(resample_eltwise_fusing_through, reshape) {
    auto p = GetParam();
    auto reshape_shape = p.out_shape;
    reshape_shape.feature[0] *= reshape_shape.spatial[0];
    reshape_shape.spatial[0] = 1;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(layout{ p.default_type, p.default_format, tensor{ 1, 1, 1, 1 } })),
        resample("resample_prim", "input", p.out_shape, p.in_shape.feature[0], p.type),
        reshape("reshape", "resample_prim", reshape_shape),
        eltwise("eltwise", "reshape", "eltwise_data", eltwise_mode::prod),
        reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p, {{"resample_prim", {"eltwise"}}});
}

#define RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP 2, 3
#define RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_INT 2, 4
INSTANTIATE_TEST_SUITE_P(fusings_gpu, resample_eltwise_fusing_through, ::testing::ValuesIn(std::vector<resample_test_params>{
    resample_test_params{ CASE_RESAMPLE_FP32_1, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP32_2, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP32_3, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP32_4, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP32_5, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP32_6, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP32_7, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP32_8, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },

    resample_test_params{ CASE_RESAMPLE_FP16_1, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP16_2, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP16_3, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP16_4, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP16_5, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP16_6, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP16_7, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },
    resample_test_params{ CASE_RESAMPLE_FP16_8, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_FP },

    resample_test_params{ CASE_RESAMPLE_I8_1, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_INT },
    resample_test_params{ CASE_RESAMPLE_I8_2, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_INT },
    resample_test_params{ CASE_RESAMPLE_I8_3, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_INT },
    resample_test_params{ CASE_RESAMPLE_I8_4, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_INT },

    resample_test_params{ CASE_RESAMPLE_U8_1, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_INT },
    resample_test_params{ CASE_RESAMPLE_U8_2, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_INT },
    resample_test_params{ CASE_RESAMPLE_U8_3, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_INT },
    resample_test_params{ CASE_RESAMPLE_U8_4, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT_INT },
}));

class resample_eltwise_fusing_through_not_allowed : public ResamplePrimitiveFusingTest {};
TEST_P(resample_eltwise_fusing_through_not_allowed, reshape_two_users) {
    auto p = GetParam();
    auto reshape_shape = p.out_shape;
    reshape_shape.feature[0] *= reshape_shape.spatial[0];
    reshape_shape.spatial[0] = 1;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(layout{ p.default_type, p.default_format, tensor{ 1, 1, 1, 1 } })),
        resample("resample_prim", "input", p.out_shape, p.in_shape.feature[0], p.type),
        reshape("reshape", "resample_prim", reshape_shape),
        eltwise("eltwise", "reshape", "eltwise_data", eltwise_mode::prod),
        eltwise("sum", "reshape", "eltwise", eltwise_mode::sum),
        reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

#define RESAMPLE_ELTWISE_FUSING_THROUGH_CNT 4, 4
INSTANTIATE_TEST_SUITE_P(fusings_gpu, resample_eltwise_fusing_through_not_allowed, ::testing::ValuesIn(std::vector<resample_test_params>{
    resample_test_params{ CASE_RESAMPLE_FP32_1, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT },
    resample_test_params{ CASE_RESAMPLE_FP32_2, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT },

    resample_test_params{ CASE_RESAMPLE_FP16_1, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT },
    resample_test_params{ CASE_RESAMPLE_FP16_2, RESAMPLE_ELTWISE_FUSING_THROUGH_CNT },
}));
