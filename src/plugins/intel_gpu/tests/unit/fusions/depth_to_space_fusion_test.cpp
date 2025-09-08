// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/depth_to_space.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct depth_to_space_test_params {
    tensor input_size;
    tensor output_size;
    depth_to_space_mode mode;
    data_types input_type;
    format input_format;
    size_t block_size;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class DepthToSpaceFusingsTest : public ::BaseFusingTest<depth_to_space_test_params> {
public:
    void execute(depth_to_space_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(depth_to_space_test_params& p) {
        return layout{ p.input_type, p.input_format, p.input_size };
    }

    layout get_per_channel_layout(depth_to_space_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.output_size.feature[0], 1, 1 } };
    }

    format get_input_format(depth_to_space_test_params &p) {
        return p.input_format;
    }
};

}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* -------------------------------- DepthToSpace cases ------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_DEPTH_TO_SPACE_F32_1 { 1, 16, 8, 10 }, { 1, 4, 16, 20 }, depth_to_space_mode::blocks_first, data_types::f32, format::bfyx, 2, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_F32_2 { 1, 32, 8, 8 },  { 1, 2, 32, 32 }, depth_to_space_mode::blocks_first, data_types::f32, format::b_fs_yx_fsv16, 4, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_F16_1 { 1, 12, 8, 8 },  { 1, 3, 16, 16 }, depth_to_space_mode::blocks_first, data_types::f16, format::bfyx, 2, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_F16_2 { 1, 16, 9, 8 },  { 1, 1, 36, 32 }, depth_to_space_mode::blocks_first, data_types::f16, format::b_fs_yx_fsv16, 4, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_U8_1  { 1, 128, 8, 8 }, { 1, 2, 64, 64 }, depth_to_space_mode::blocks_first, data_types::u8, format::bfyx, 8, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_U8_2  { 1, 128, 4, 8 }, { 1, 8, 16, 32 }, depth_to_space_mode::blocks_first, data_types::u8, format::b_fs_yx_fsv16, 4, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_I8_1  { 1, 16, 8, 8 },  { 1, 4, 16, 16 }, depth_to_space_mode::blocks_first, data_types::i8, format::bfyx, 2, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_I8_2  { 1, 256, 8, 8 }, { 1, 4, 64, 64 }, depth_to_space_mode::blocks_first, data_types::i8, format::b_fs_yx_fsv16, 8, data_types::f32, format::bfyx

class depth_to_space_quantize_i8 : public DepthToSpaceFusingsTest {};
TEST_P(depth_to_space_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        depth_to_space("depth_to_space", input_info("input"), p.block_size, p.mode),
        data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -128)),
        data("out_high", get_mem(get_single_element_layout(p), 127)),
        quantize("quant", input_info("depth_to_space"), input_info("in_low"), input_info("in_high"),
                 input_info("out_low"), input_info("out_high"), 256, data_types::i8),
        reorder("reorder_bfyx", input_info("quant"), format::bfyx, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, depth_to_space_quantize_i8, ::testing::ValuesIn(std::vector<depth_to_space_test_params>{
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F32_1, 2, 3 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F32_2, 2, 3 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F16_1, 2, 3 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F16_2, 2, 3 },
}));

class depth_to_space_scale_act_eltwise_quantize_u8 : public DepthToSpaceFusingsTest {};
TEST_P(depth_to_space_scale_act_eltwise_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        depth_to_space("depth_to_space", input_info("input"), p.block_size, p.mode),
        data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
        eltwise("scale1", { input_info("depth_to_space"), input_info("scale1_data") }, eltwise_mode::prod, p.default_type),
        activation("actv1", input_info("scale1"), activation_func::relu),
        data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
        eltwise("eltw", { input_info("actv1"), input_info("eltw_data") }, eltwise_mode::sum, p.default_type),
        data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), 0)),
        data("out_high", get_mem(get_single_element_layout(p), 255)),
        quantize("quant", input_info("eltw"), input_info("in_low"), input_info("in_high"),
                 input_info("out_low"), input_info("out_high"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quant"), format::bfyx, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, depth_to_space_scale_act_eltwise_quantize_u8, ::testing::ValuesIn(std::vector<depth_to_space_test_params>{
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F32_1, 2, 6 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F32_2, 2, 6 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F16_1, 2, 6 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F16_2, 2, 6 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_U8_1, 2, 6 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_U8_2, 2, 6 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_I8_1, 2, 6 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_I8_2, 2, 6 },
}));


class depth_to_space_scale_act_eltw : public DepthToSpaceFusingsTest {};
TEST_P(depth_to_space_scale_act_eltw, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        depth_to_space("depth_to_space", input_info("input"), p.block_size, p.mode),
        data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
        eltwise("scale1", { input_info("depth_to_space"), input_info("scale1_data") }, eltwise_mode::prod, p.default_type),
        activation("actv1", input_info("scale1"), activation_func::relu),
        data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
        eltwise("eltw", { input_info("actv1"), input_info("eltw_data") }, eltwise_mode::sum, p.default_type),
        reorder("reorder_bfyx", input_info("eltw"), format::bfyx, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, depth_to_space_scale_act_eltw, ::testing::ValuesIn(std::vector<depth_to_space_test_params>{
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F32_1, 2, 5 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F32_2, 2, 5 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F16_1, 2, 5 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_F16_2, 2, 5 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_U8_1, 2, 5 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_U8_2, 2, 5 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_I8_1, 2, 5 },
    depth_to_space_test_params{ CASE_DEPTH_TO_SPACE_I8_2, 2, 5 },
}));
