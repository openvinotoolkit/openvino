// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/batch_to_space.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct batch_to_space_test_params {
    tensor input_size;
    tensor output_size;
    data_types input_type;
    format input_format;
    tensor block_shape;
    tensor crops_begin;
    tensor crops_end;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class BatchToSpaceFusingsTest : public ::BaseFusingTest<batch_to_space_test_params> {
public:
    void execute(batch_to_space_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(batch_to_space_test_params& p) {
        return layout{ p.input_type, p.input_format, p.input_size };
    }

    layout get_per_channel_layout(batch_to_space_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.output_size.feature[0], 1, 1 } };
    }
};

}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- BatchToSpace cases ----------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_BATCH_TO_SPACE_F32_1 { 8,  1,  1, 1 }, { 2, 1,   2,  2 }, data_types::f32, format::bfyx,          { 1, 1, 2,  2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_F32_2 { 64, 16, 2, 2 }, { 2, 112, 4,  4 }, data_types::f32, format::b_fs_yx_fsv16, { 1, 8, 2,  2 }, { 0, 8, 0, 0 }, { 0, 8, 0, 0 }, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_F16_1 { 16, 4,  1, 2 }, { 2, 12,  1,  2 }, data_types::f16, format::bfyx,          { 1, 4, 2,  1 }, { 0, 2, 1, 0 }, { 0, 2, 0, 0 }, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_F16_2 { 32, 16, 2, 1 }, { 1, 16,  32, 2 }, data_types::f16, format::b_fs_yx_fsv16, { 1, 1, 16, 2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_U8_1  { 30, 12, 4, 6 }, { 1, 52,  8,  9 }, data_types::u8,  format::bfyx,          { 1, 5, 2,  3 }, { 0, 8, 0, 9 }, { 0, 0, 0, 0 }, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_U8_2  { 24, 32, 4, 5 }, { 2, 64,  12, 8 }, data_types::u8,  format::b_fs_yx_fsv16, { 1, 2, 3,  2 }, { 0, 0, 0, 2 }, { 0, 0, 0, 0 }, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_I8_1  { 32, 1,  3, 4 }, { 1, 8,   6,  8 }, data_types::i8,  format::bfyx,          { 1, 8, 2,  2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_I8_2  { 16, 16, 2, 1 }, { 2, 32,  4,  2 }, data_types::i8,  format::b_fs_yx_fsv16, { 1, 2, 2,  2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, data_types::f32, format::bfyx

class batch_to_space_quantize_i8 : public BatchToSpaceFusingsTest {};
TEST_P(batch_to_space_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        batch_to_space("batch_to_space", input_info("input"), p.block_shape, p.crops_begin, p.crops_end, p.output_size),
        data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -128)),
        data("out_high", get_mem(get_single_element_layout(p), 127)),
        quantize("quant", input_info("batch_to_space"), input_info("in_low"), input_info("in_high"),
                 input_info("out_low"), input_info("out_high"), 256, data_types::i8),
        reorder("reorder_bfyx", input_info("quant"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, batch_to_space_quantize_i8, ::testing::ValuesIn(std::vector<batch_to_space_test_params>{
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F32_1, 2, 3 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F32_2, 2, 3 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F16_1, 2, 3 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F16_2, 2, 3 },
}));

class batch_to_space_scale_act_eltwise_quantize_u8 : public BatchToSpaceFusingsTest {};
TEST_P(batch_to_space_scale_act_eltwise_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        batch_to_space("batch_to_space", input_info("input"), p.block_shape, p.crops_begin, p.crops_end, p.output_size),
        data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
        eltwise("scale1", { input_info("batch_to_space"), input_info("scale1_data") }, eltwise_mode::prod, p.default_type),
        activation("actv1", input_info("scale1"), activation_func::relu),
        data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
        eltwise("eltw", { input_info("actv1"), input_info("eltw_data") }, eltwise_mode::sum, p.default_type),
        data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), 0)),
        data("out_high", get_mem(get_single_element_layout(p), 255)),
        quantize("quant", input_info("eltw"), input_info("in_low"), input_info("in_high"),
                 input_info("out_low"), input_info("out_high"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quant"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, batch_to_space_scale_act_eltwise_quantize_u8, ::testing::ValuesIn(std::vector<batch_to_space_test_params>{
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F32_1, 2, 6 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F32_2, 2, 6 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F16_1, 2, 6 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F16_2, 2, 6 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_U8_1, 2, 6 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_U8_2, 2, 6 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_I8_1, 2, 6 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_I8_2, 2, 6 },
}));

class batch_to_space_scale_act_eltw : public BatchToSpaceFusingsTest {};
TEST_P(batch_to_space_scale_act_eltw, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        batch_to_space("batch_to_space", input_info("input"), p.block_shape, p.crops_begin, p.crops_end, p.output_size),
        data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
        eltwise("scale1", { input_info("batch_to_space"), input_info("scale1_data") }, eltwise_mode::prod, p.default_type),
        activation("actv1", input_info("scale1"), activation_func::relu),
        data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
        eltwise("eltw", { input_info("actv1"), input_info("eltw_data") }, eltwise_mode::sum, p.default_type),
        reorder("reorder_bfyx", input_info("eltw"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, batch_to_space_scale_act_eltw, ::testing::ValuesIn(std::vector<batch_to_space_test_params>{
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F32_1, 2, 5 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F32_2, 2, 5 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F16_1, 2, 5 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_F16_2, 2, 5 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_U8_1, 2, 5 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_U8_2, 2, 5 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_I8_1, 2, 5 },
    batch_to_space_test_params{ CASE_BATCH_TO_SPACE_I8_2, 2, 5 },
}));
