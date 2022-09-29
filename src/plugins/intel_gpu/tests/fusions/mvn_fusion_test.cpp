// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/mvn.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct mvn_test_params {
    tensor input_size;
    tensor elwise_size;
    data_types input_type;
    format input_format;
    bool across_channels;
    bool normalize_variance;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class MVNFusingTest : public ::BaseFusingTest<mvn_test_params> {
public:
    void execute(mvn_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(mvn_test_params& p) {
        return layout{ p.input_type, p.input_format, p.input_size };
    }

    layout get_per_channel_layout(mvn_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.input_size.feature[0], 1, 1 } };
    }
};
}  // namespace


/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------------- MVN cases --------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_MVN_F32_1      { 1, 16, 8, 8 },    { 1, 16, 8, 8 },    data_types::f32, format::bfyx, false, true, data_types::f32, format::bfyx
#define CASE_MVN_F32_2      { 2, 16, 8, 8 },    { 2, 16, 8, 8 },    data_types::f32, format::bfyx, true, true, data_types::f32, format::bfyx
#define CASE_MVN_3D_F32_1   { 1, 16, 8, 8, 8 }, { 1, 16, 8, 8, 8 }, data_types::f32, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_F32_2   { 2, 16, 8, 8, 8 }, { 2, 16, 8, 8, 8 }, data_types::f32, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_F32_3   { 2, 8, 4, 4, 4 },  { 2, 8, 1, 1, 1 },  data_types::f32, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_F16_1      { 1, 16, 8, 8 },    { 1, 16, 8, 8 },    data_types::f16, format::bfyx, false, true, data_types::f16, format::bfyx
#define CASE_MVN_F16_2      { 2, 16, 8, 8 },    { 2, 16, 8, 8 },    data_types::f16, format::bfyx, true, true, data_types::f16, format::bfyx
#define CASE_MVN_3D_F16_1   { 1, 16, 8, 8, 8 }, { 1, 16, 8, 8, 8 }, data_types::f16, format::bfzyx, false, true, data_types::f16, format::bfzyx
#define CASE_MVN_3D_F16_2   { 2, 16, 8, 8, 8 }, { 2, 16, 8, 8, 8 }, data_types::f16, format::bfzyx, true, true, data_types::f16, format::bfzyx
#define CASE_MVN_I8_1       { 1, 16, 8, 8 },    { 1, 16, 8, 8 },    data_types::i8, format::bfyx, false, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_2       { 2, 16, 8, 8 },    { 2, 16, 8, 8 },    data_types::i8, format::bfyx, true, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_3       { 1, 16, 8, 8 },    { 1, 16, 8, 8 },    data_types::i8, format::b_fs_yx_fsv16, false, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_4       { 2, 16, 8, 8 },    { 2, 16, 8, 8 },    data_types::i8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_5       { 2, 16, 8, 8 },    { 1, 1, 1, 8 },     data_types::i8, format::b_fs_yx_fsv16, false, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_6       { 2, 16, 8, 8 },    { 1, 1, 1, 1 },     data_types::i8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_7       { 2, 16, 1, 8 },    { 1, 1, 8, 1 },     data_types::i8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_3D_I8_1    { 1, 16, 8, 8, 8 }, { 1, 16, 8, 8, 8 }, data_types::i8, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_I8_2    { 2, 16, 8, 8, 8 }, { 2, 16, 8, 8, 8 }, data_types::i8, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_I8_3    { 2, 16, 8, 8, 8 }, { 2, 1, 8, 8, 1 },  data_types::i8, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_I8_4    { 2, 16, 8, 8, 8 }, { 2, 16, 8, 1, 8 }, data_types::i8, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_I8_5    { 2, 2, 1, 2, 1 },  { 2, 2, 2, 2, 2 },  data_types::i8, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_U8_1       { 1, 16, 8, 8 },    { 1, 16, 8, 8 },    data_types::u8, format::bfyx, false, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_2       { 2, 16, 8, 8 },    { 2, 16, 8, 8 },    data_types::u8, format::bfyx, true, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_3       { 1, 16, 8, 8 },    { 1, 16, 8, 8 },    data_types::u8, format::b_fs_yx_fsv16, false, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_4       { 2, 16, 8, 8 },    { 2, 16, 8, 8 },    data_types::u8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_5       { 2, 16, 8, 8 },    { 2, 1, 8, 8 },     data_types::u8, format::b_fs_yx_fsv16, false, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_6       { 2, 16, 8, 8 },    { 1, 1, 1, 8 },     data_types::u8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_7       { 1, 16, 16, 1 },   { 1, 16, 1, 16 },   data_types::u8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_3D_U8_1    { 1, 16, 8, 8, 8 }, { 1, 16, 8, 8, 8 }, data_types::u8, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_U8_2    { 2, 16, 8, 8, 8 }, { 2, 16, 8, 8, 8 }, data_types::u8, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_U8_3    { 2, 16, 8, 8, 8 }, { 2, 1, 1, 1, 1 },  data_types::u8, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_U8_4    { 2, 16, 8, 8, 8 }, { 1, 1, 1, 1, 1 },  data_types::u8, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_U8_5    { 2, 16, 1, 8, 8 }, { 1, 1, 8, 1, 1 },  data_types::u8, format::bfzyx, false, true, data_types::f32, format::bfzyx

class mvn_activation : public MVNFusingTest {};
TEST_P(mvn_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        mvn("mvn", "input", p.normalize_variance, 1e-10f, false, false),
        activation("act", "mvn", activation_func::hyperbolic_tan),
        reorder("reorder_bfyx", "act", format::bfyx, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, mvn_activation, ::testing::ValuesIn(std::vector<mvn_test_params>{
    mvn_test_params{ CASE_MVN_F32_1, 2, 3 },
    mvn_test_params{ CASE_MVN_F32_2, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_F32_1, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_F32_2, 2, 3 },
    mvn_test_params{ CASE_MVN_F16_1, 2, 3 },
    mvn_test_params{ CASE_MVN_F16_2, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_F16_1, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_F16_2, 2, 3 },
    mvn_test_params{ CASE_MVN_I8_1, 2, 3 },
    mvn_test_params{ CASE_MVN_I8_2, 2, 3 },
    mvn_test_params{ CASE_MVN_I8_3, 2, 3 },
    mvn_test_params{ CASE_MVN_I8_4, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_I8_1, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_I8_2, 2, 3 },
    mvn_test_params{ CASE_MVN_U8_1, 2, 3 },
    mvn_test_params{ CASE_MVN_U8_2, 2, 3 },
    mvn_test_params{ CASE_MVN_U8_3, 2, 3 },
    mvn_test_params{ CASE_MVN_U8_4, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_U8_1, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_U8_2, 2, 3 },
}));

class mvn_scale_quantize_i8 : public MVNFusingTest {};
TEST_P(mvn_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        mvn("mvn", "input", p.normalize_variance, 1e-10f, false, false),
        data("scale_data", get_mem(get_per_channel_layout(p))),
        eltwise("scale", { "mvn", "scale_data" }, eltwise_mode::prod, p.default_type),
        data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -127, 127)),
        data("out_high", get_mem(get_single_element_layout(p), -127, 127)),
        quantize("quant", "scale", "in_low", "in_high", "out_low", "out_high", 255, data_types::i8),
        reorder("reorder_bfyx", "quant", format::bfyx, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, mvn_scale_quantize_i8, ::testing::ValuesIn(std::vector<mvn_test_params>{
    // Full fusing for fp input not supported yet, it may lead to output padding and non-optimal kernel
    // mvn_test_params{ CASE_MVN_F32_1, 2, 4 },
    // mvn_test_params{ CASE_MVN_F32_2, 2, 4 },
    // mvn_test_params{ CASE_MVN_3D_F32_1, 2, 4 },
    // mvn_test_params{ CASE_MVN_3D_F32_2, 2, 4 },
    // mvn_test_params{ CASE_MVN_F16_1, 2, 4 },
    // mvn_test_params{ CASE_MVN_F16_2, 2, 4 },
    // mvn_test_params{ CASE_MVN_3D_F16_1, 2, 4 },
    // mvn_test_params{ CASE_MVN_3D_F16_2, 2, 4 },
    mvn_test_params{ CASE_MVN_I8_1, 2, 4 },
    mvn_test_params{ CASE_MVN_I8_2, 2, 4 },
    mvn_test_params{ CASE_MVN_I8_3, 2, 4 },
    mvn_test_params{ CASE_MVN_I8_4, 2, 4 },
    mvn_test_params{ CASE_MVN_3D_I8_1, 2, 4 },
    mvn_test_params{ CASE_MVN_3D_I8_2, 2, 4 },
    mvn_test_params{ CASE_MVN_U8_1, 2, 4 },
    mvn_test_params{ CASE_MVN_U8_2, 2, 4 },
    mvn_test_params{ CASE_MVN_U8_3, 2, 4 },
    mvn_test_params{ CASE_MVN_U8_4, 2, 4 },
    mvn_test_params{ CASE_MVN_3D_U8_1, 2, 4 },
    mvn_test_params{ CASE_MVN_3D_U8_2, 2, 4 },
}));

class mvn_scale_activation_eltwise_fp32_quantize_i8 : public MVNFusingTest {};
TEST_P(mvn_scale_activation_eltwise_fp32_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        mvn("mvn", "input", p.normalize_variance, 1e-10f, false, false),
        data("scale_data", get_mem(get_per_channel_layout(p))),
        eltwise("scale", { "mvn", "scale_data" }, eltwise_mode::prod, p.default_type),
        activation("act", "scale", activation_func::hyperbolic_tan),
        data("eltw_data", get_mem(layout{ p.input_type, p.default_format, p.elwise_size })),
        eltwise("eltw", { "act", "eltw_data" }, eltwise_mode::sum, data_types::f32),
        data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -128)),
        data("out_high", get_mem(get_single_element_layout(p), 127)),
        quantize("quant", "eltw", "in_low", "in_high", "out_low", "out_high", 256, data_types::i8),
        reorder("reorder_bfyx", "quant", format::bfyx, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, mvn_scale_activation_eltwise_fp32_quantize_i8, ::testing::ValuesIn(std::vector<mvn_test_params>{
    // Full using for fp input not supported yet, it may lead to output padding and non-optimal kernel
    // mvn_test_params{ CASE_MVN_F32_1, 2, 7 },
    // mvn_test_params{ CASE_MVN_F32_2, 2, 7 },
    // mvn_test_params{ CASE_MVN_3D_F32_1, 2, 7 },
    // mvn_test_params{ CASE_MVN_3D_F32_2, 2, 7 },
    // mvn_test_params{ CASE_MVN_F16_1, 2, 7 },
    // mvn_test_params{ CASE_MVN_F16_2, 2, 7 },
    // mvn_test_params{ CASE_MVN_3D_F16_1, 2, 7 },
    // mvn_test_params{ CASE_MVN_3D_F16_2, 2, 7 },
    mvn_test_params{ CASE_MVN_I8_1, 2, 6 },
    mvn_test_params{ CASE_MVN_I8_2, 2, 6 },
    mvn_test_params{ CASE_MVN_I8_3, 2, 6 },
    mvn_test_params{ CASE_MVN_I8_4, 2, 6 },
    mvn_test_params{ CASE_MVN_I8_5, 2, 6 },
    mvn_test_params{ CASE_MVN_I8_6, 2, 6 },
    mvn_test_params{ CASE_MVN_I8_7, 3, 6 },
    mvn_test_params{ CASE_MVN_3D_I8_1, 2, 6 },
    mvn_test_params{ CASE_MVN_3D_I8_2, 2, 6 },
    mvn_test_params{ CASE_MVN_3D_I8_3, 2, 6 },
    mvn_test_params{ CASE_MVN_3D_I8_4, 2, 6 },
    mvn_test_params{ CASE_MVN_3D_I8_5, 3, 6 },
    mvn_test_params{ CASE_MVN_U8_1, 2, 6 },
    mvn_test_params{ CASE_MVN_U8_2, 2, 6 },
    mvn_test_params{ CASE_MVN_U8_3, 2, 6 },
    mvn_test_params{ CASE_MVN_U8_4, 2, 6 },
    mvn_test_params{ CASE_MVN_U8_5, 2, 6 },
    mvn_test_params{ CASE_MVN_U8_6, 2, 6 },
    mvn_test_params{ CASE_MVN_U8_7, 3, 6 },
    mvn_test_params{ CASE_MVN_3D_U8_1, 2, 6 },
    mvn_test_params{ CASE_MVN_3D_U8_2, 2, 6 },
    mvn_test_params{ CASE_MVN_3D_U8_3, 2, 6 },
    mvn_test_params{ CASE_MVN_3D_U8_4, 2, 6 },
    mvn_test_params{ CASE_MVN_3D_U8_5, 3, 6 },
}));

class mvn_eltwise : public MVNFusingTest {};
TEST_P(mvn_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", layout{ p.input_type, p.input_format, p.input_size }),
        mvn("mvn", "input", p.normalize_variance, 1e-10f, false, false),
        data("eltw_data", get_mem(layout{ p.input_type, p.default_format, p.elwise_size })),
        eltwise("eltw", { "mvn", "eltw_data" }, eltwise_mode::sum, data_types::f32),
        reorder("reorder_bfyx", "eltw", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, mvn_eltwise, ::testing::ValuesIn(std::vector<mvn_test_params>{
    mvn_test_params{ CASE_MVN_I8_5, 2, 3 },
    mvn_test_params{ CASE_MVN_I8_6, 2, 3 },
    mvn_test_params{ CASE_MVN_I8_7, 3, 3 },
    mvn_test_params{ CASE_MVN_3D_I8_3, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_I8_4, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_I8_5, 3, 3 },
    mvn_test_params{ CASE_MVN_U8_1, 2, 3 },
    mvn_test_params{ CASE_MVN_U8_2, 2, 3 },
    mvn_test_params{ CASE_MVN_U8_3, 2, 3 },
    mvn_test_params{ CASE_MVN_U8_4, 2, 3 },
    mvn_test_params{ CASE_MVN_U8_5, 2, 3 },
    mvn_test_params{ CASE_MVN_U8_6, 2, 3 },
    mvn_test_params{ CASE_MVN_U8_7, 3, 3 },
    mvn_test_params{ CASE_MVN_3D_U8_1, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_U8_2, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_U8_3, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_U8_4, 2, 3 },
    mvn_test_params{ CASE_MVN_3D_U8_5, 3, 3 },
}));

class mvn_eltwise_f16 : public MVNFusingTest {};
TEST_P(mvn_eltwise_f16, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", layout{ p.input_type, p.input_format, p.input_size }),
        mvn("mvn", "input", p.normalize_variance, 1e-10f, false, false),
        data("eltw_data", get_mem(layout{ p.input_type, p.default_format, p.elwise_size })),
        eltwise("eltw", { "mvn", "eltw_data" }, eltwise_mode::sum, data_types::f16),
        reorder("reorder_bfyx", "eltw", p.default_format, data_types::f32)
    );

    tolerance = 0.1f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, mvn_eltwise_f16, ::testing::ValuesIn(std::vector<mvn_test_params>{
    mvn_test_params{ CASE_MVN_I8_6, 2, 3 },
    mvn_test_params{ CASE_MVN_U8_2, 2, 3 },
}));
