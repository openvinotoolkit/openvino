// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gather_nd.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct gather_nd_test_params {
    data_types data_type;

    format input_format;
    tensor input_shape;

    format indices_format;
    tensor indices_shape;

    format output_format;
    tensor output_shape;

    int max_number_in_indices;
    int indices_rank;
    int batch_dims;

    data_types default_type;
    format default_format;

    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class GatherNDPrimitiveFusingTest : public ::BaseFusingTest<gather_nd_test_params> {
public:
    void execute(gather_nd_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(gather_nd_test_params& p) {
        return layout{ p.data_type, p.input_format, p.input_shape };
    }

    layout get_indices_layout(gather_nd_test_params& p) {
        return layout{ p.data_type, p.indices_format, p.indices_shape };
    }

    layout get_output_layout(gather_nd_test_params& p) {
        return layout{ p.data_type, p.output_format, p.output_shape };
    }

    layout get_per_channel_layout(gather_nd_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.output_shape.feature[0], 1, 1 } };
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ GatherND cases ------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
#define CASE_GATHER_ND_FP16_4D_1 data_types::f16, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 3, 1, 1, 1 }, format::bfyx, { 3, 7, 9, 8 }, 6, 2, 0, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_4D_2 data_types::f16, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 6, 1, 1, 1 }, format::bfyx, { 6, 8, 1, 9 }, 6, 2, 1, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_4D_3 data_types::f16, format::bfyx, { 5, 4, 7, 2 }, format::bfyx, { 5, 4, 1, 2 }, format::bfyx, { 40, 1, 1, 1 }, 6, 4, 3, data_types::f16, format::bfyx

#define CASE_GATHER_ND_FP16_5D_1 data_types::f16, format::bfzyx, { 5, 6, 7, 8, 5 }, format::bfyx, { 5, 1, 1, 1 }, format::bfzyx, { 5, 6, 7, 8, 5 }, 5, 2, 0, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_5D_2 data_types::f16, format::bfzyx, { 5, 6, 7, 8, 5 }, format::bfyx, { 5, 1, 1, 1 }, format::bfyx, { 5, 5, 7, 8 }, 5, 2, 1, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_5D_3 data_types::f16, format::bfzyx, { 5, 4, 7, 8, 5 }, format::bfyx, { 5, 4, 1, 3 }, format::bfyx, { 20, 1, 1, 1 }, 4, 3, 2, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_5D_4 data_types::f16, format::bfzyx, { 5, 4, 7, 8, 3 }, format::bfyx, { 5, 4, 1, 3 }, format::bfyx, { 60, 7, 1, 1 }, 4, 4, 3, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_5D_5 data_types::f16, format::bfzyx, { 5, 4, 7, 2, 3 }, format::bfzyx, { 5, 4, 1, 2, 3 }, format::bfyx, { 120, 1, 1, 1 }, 4, 5, 4, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_5D_6 data_types::f16, format::bfzyx, { 5, 4, 7, 4, 4 }, format::bfzyx, { 5, 4, 1, 1, 3 }, format::bfzyx, { 20, 3, 7, 4, 1 }, 4, 5, 2, data_types::f16, format::bfyx

#define CASE_GATHER_ND_FP16_6D_1 data_types::f16, format::bfwzyx, { 5, 4, 6, 7, 8, 5 }, format::bfyx, { 5, 4, 2, 2 }, format::bfyx, { 20, 2, 6, 7 }, 5, 4, 2, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_6D_2 data_types::f16, format::bfwzyx, { 5, 4, 6, 7, 8, 2 }, format::bfyx, { 5, 4, 2, 2 }, format::bfyx, { 40, 6, 1, 1 }, 5, 4, 3, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_6D_3 data_types::f16, format::bfwzyx, { 5, 4, 6, 7, 2, 2 }, format::bfzyx, { 5, 4, 1, 2, 2 }, format::bfyx, { 80, 6, 1, 1 }, 5, 5, 4, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_6D_4 data_types::f16, format::bfwzyx, { 5, 4, 6, 3, 2, 2 }, format::bfwzyx, { 5, 4, 1, 3, 2, 2 }, format::bfyx, { 240, 1, 1, 1 }, 5, 6, 5, data_types::f16, format::bfyx

#define CASE_GATHER_ND_FP32_4D_1 data_types::f32, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 3, 1, 1, 1 }, format::bfyx, { 3, 7, 9, 8 }, 6, 2, 0, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_4D_2 data_types::f32, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 6, 1, 1, 1 }, format::bfyx, { 6, 8, 1, 9 }, 6, 2, 1, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_4D_3 data_types::f32, format::bfyx, { 5, 4, 7, 2 }, format::bfyx, { 5, 4, 1, 2 }, format::bfyx, { 40, 1, 1, 1 }, 6, 4, 3, data_types::f32, format::bfyx

#define CASE_GATHER_ND_FP32_5D_1 data_types::f32, format::bfzyx, { 5, 6, 7, 8, 5 }, format::bfyx, { 5, 1, 1, 1 }, format::bfzyx, { 5, 6, 7, 8, 5 }, 5, 2, 0, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_5D_2 data_types::f32, format::bfzyx, { 5, 6, 7, 8, 5 }, format::bfyx, { 5, 1, 1, 1 }, format::bfyx, { 5, 5, 7, 8 }, 5, 2, 1, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_5D_3 data_types::f32, format::bfzyx, { 5, 4, 7, 8, 5 }, format::bfyx, { 5, 4, 1, 3 }, format::bfyx, { 20, 1, 1, 1 }, 4, 3, 2, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_5D_4 data_types::f32, format::bfzyx, { 5, 4, 7, 8, 3 }, format::bfyx, { 5, 4, 1, 3 }, format::bfyx, { 60, 7, 1, 1 }, 4, 4, 3, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_5D_5 data_types::f32, format::bfzyx, { 5, 4, 7, 2, 3 }, format::bfzyx, { 5, 4, 1, 2, 3 }, format::bfyx, { 120, 1, 1, 1 }, 4, 5, 4, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_5D_6 data_types::f32, format::bfzyx, { 5, 4, 7, 4, 4 }, format::bfzyx, { 5, 4, 1, 1, 3 }, format::bfzyx, { 20, 3, 7, 4, 1 }, 4, 5, 2, data_types::f32, format::bfyx

#define CASE_GATHER_ND_FP32_6D_1 data_types::f32, format::bfwzyx, { 5, 4, 6, 7, 8, 5 }, format::bfyx, { 5, 4, 2, 2 }, format::bfyx, { 20, 2, 6, 7 }, 5, 4, 2, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_6D_2 data_types::f32, format::bfwzyx, { 5, 4, 6, 7, 8, 2 }, format::bfyx, { 5, 4, 2, 2 }, format::bfyx, { 40, 6, 1, 1 }, 5, 4, 3, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_6D_3 data_types::f32, format::bfwzyx, { 5, 4, 6, 7, 2, 2 }, format::bfzyx, { 5, 4, 1, 2, 2 }, format::bfyx, { 80, 6, 1, 1 }, 5, 5, 4, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_6D_4 data_types::f32, format::bfwzyx, { 5, 4, 6, 3, 2, 2 }, format::bfwzyx, { 5, 4, 1, 3, 2, 2 }, format::bfyx, { 240, 1, 1, 1 }, 5, 6, 5, data_types::f32, format::bfyx

class gather_nd_quantize : public GatherNDPrimitiveFusingTest {};
TEST_P(gather_nd_quantize, basic) {
    auto p = GetParam();

    auto input_rank = 0;
    if (p.input_format == format::bfyx) {
        input_rank = 4;
    } else if (p.input_format == format::bfzyx) {
        input_rank = 5;
    } else if (p.input_format == format::bfwzyx) {
        input_rank = 6;
    }

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("gather_nd_indices", get_mem(get_indices_layout(p), 0, p.max_number_in_indices - 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gather_nd("gather_nd_prim", input_info("input"), input_info("gather_nd_indices"), input_rank, p.indices_rank, p.batch_dims),
        quantize("quantize", input_info("gather_nd_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_nd_quantize, ::testing::ValuesIn(std::vector<gather_nd_test_params>{
    gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_1, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_2, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_3, 2, 3 },

    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_1, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_2, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_3, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_4, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_5, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_6, 2, 3 },

    gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_1, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_2, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_3, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_4, 2, 3 },

    gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_1, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_2, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_3, 2, 3 },

    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_1, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_2, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_3, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_4, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_5, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_6, 2, 3 },

    gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_1, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_2, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_3, 2, 3 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_4, 2, 3 },
}));

class gather_nd_activation_scale_eltwise : public GatherNDPrimitiveFusingTest {};
TEST_P(gather_nd_activation_scale_eltwise, basic) {
    auto p = GetParam();

    auto input_rank = 0;
    if (p.input_format == format::bfyx) {
        input_rank = 4;
    } else if (p.input_format == format::bfzyx) {
        input_rank = 5;
    } else if (p.input_format == format::bfwzyx) {
        input_rank = 6;
    }

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("gather_nd_indices", get_mem(get_indices_layout(p), 0, p.max_number_in_indices - 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 255)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        gather_nd("gather_nd_prim", input_info("input"), input_info("gather_nd_indices"), input_rank, p.indices_rank, p.batch_dims),
        activation("activation", input_info("gather_nd_prim"), activation_func::abs),
        eltwise("scale", { input_info("activation"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        eltwise("eltwise", { input_info("scale"), input_info("eltwise_data") }, eltwise_mode::sum, p.data_type),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_nd_activation_scale_eltwise, ::testing::ValuesIn(std::vector<gather_nd_test_params>{
    gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_1, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_2, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_3, 2, 5 },

    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_1, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_2, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_3, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_4, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_5, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_6, 2, 5 },

    gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_1, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_2, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_3, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_4, 2, 5 },

    gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_1, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_2, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_3, 2, 5 },

    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_1, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_2, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_3, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_4, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_5, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_6, 2, 5 },

    gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_1, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_2, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_3, 2, 5 },
    gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_4, 2, 5 },
}));
