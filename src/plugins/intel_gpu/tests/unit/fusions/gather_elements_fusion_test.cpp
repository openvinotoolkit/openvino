// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gather_elements.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct gather_elements_test_params {
    data_types data_type;

    format input_format;
    tensor input_shape;

    format indices_format;
    tensor indices_shape;

    format output_format;
    tensor output_shape;

    int64_t axis;

    data_types default_type;
    format default_format;

    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class GatherElementsPrimitiveFusingTest : public ::BaseFusingTest<gather_elements_test_params> {
public:
    void execute(gather_elements_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    size_t get_axis_dim(gather_elements_test_params& p) {
        size_t rank = p.input_format.dimension();
        auto default_fmt = format::get_default_format(rank);
        auto dims = p.input_shape.sizes(default_fmt);
        return dims[p.axis];
    }

    layout get_input_layout(gather_elements_test_params& p) {
        return layout{ p.data_type, p.input_format, p.input_shape };
    }

    layout get_indices_layout(gather_elements_test_params& p) {
        return layout{ p.data_type, p.indices_format, p.indices_shape };
    }

    layout get_output_layout(gather_elements_test_params& p) {
        return layout{ p.data_type, p.output_format, p.output_shape };
    }

    layout get_per_channel_layout(gather_elements_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.output_shape.feature[0], 1, 1 } };
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ GatherElements cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
#define CASE_GATHER_ELEMENTS_FP16_4D_1 data_types::f16, format::bfyx, { 3, 7, 9, 8 }, format::bfyx, { 3, 7, 9, 8 }, format::bfyx, { 3, 7, 9, 8 }, 2, data_types::f16, format::bfyx
#define CASE_GATHER_ELEMENTS_FP16_4D_2 data_types::f16, format::bfyx, { 3, 2, 8, 3 }, format::bfyx, { 2, 2, 8, 3 }, format::bfyx, { 2, 2, 8, 3 }, 0, data_types::f16, format::bfyx
#define CASE_GATHER_ELEMENTS_FP16_4D_3 data_types::f16, format::bfyx, { 1, 3, 2, 9 }, format::bfyx, { 1, 3, 5, 9 }, format::bfyx, { 1, 3, 5, 9 }, 3, data_types::f16, format::bfyx

#define CASE_GATHER_ELEMENTS_FP16_5D_1 data_types::f16, format::bfzyx, { 3, 2, 5, 2, 3 }, format::bfzyx, { 3, 2, 2, 2, 3 }, format::bfzyx, { 3, 2, 2, 2, 3 }, 4, data_types::f16, format::bfzyx
#define CASE_GATHER_ELEMENTS_FP16_5D_2 data_types::f16, format::bfzyx, { 5, 4, 7, 4, 4 }, format::bfzyx, { 5, 4, 7, 4, 3 }, format::bfzyx, { 5, 4, 7, 4, 3 }, 2, data_types::f16, format::bfzyx

#define CASE_GATHER_ELEMENTS_FP16_6D_1 data_types::f16, format::bfwzyx, { 5, 4, 6, 7, 8, 2 }, format::bfwzyx, { 5, 2, 6, 7, 8, 2 }, format::bfwzyx, { 5, 2, 6, 7, 8, 2 }, 1, data_types::f16, format::bfwzyx
#define CASE_GATHER_ELEMENTS_FP16_6D_2 data_types::f16, format::bfwzyx, { 2, 1, 2, 3, 2, 1 }, format::bfwzyx, { 2, 1, 2, 3, 2, 3 }, format::bfwzyx, { 2, 1, 2, 3, 2, 3 }, 2, data_types::f16, format::bfwzyx
#define CASE_GATHER_ELEMENTS_FP16_6D_3 data_types::f16, format::bfwzyx, { 2, 2, 3, 4, 4, 2 }, format::bfwzyx, { 2, 2, 6, 4, 4, 2 }, format::bfwzyx, { 2, 2, 6, 4, 4, 2 }, 5, data_types::f16, format::bfwzyx


#define CASE_GATHER_ELEMENTS_FP32_4D_1 data_types::f32, format::bfyx, { 3, 7, 9, 8 }, format::bfyx, { 3, 7, 9, 8 }, format::bfyx, { 3, 7, 9, 8 }, 2, data_types::f32, format::bfyx
#define CASE_GATHER_ELEMENTS_FP32_4D_2 data_types::f32, format::bfyx, { 3, 2, 8, 3 }, format::bfyx, { 2, 2, 8, 3 }, format::bfyx, { 2, 2, 8, 3 }, 0, data_types::f32, format::bfyx
#define CASE_GATHER_ELEMENTS_FP32_4D_3 data_types::f32, format::bfyx, { 1, 3, 2, 9 }, format::bfyx, { 1, 3, 5, 9 }, format::bfyx, { 1, 3, 5, 9 }, 3, data_types::f32, format::bfyx

#define CASE_GATHER_ELEMENTS_FP32_5D_1 data_types::f32, format::bfzyx, { 3, 2, 5, 2, 3 }, format::bfzyx, { 3, 2, 2, 2, 3 }, format::bfzyx, { 3, 2, 2, 2, 3 }, 4, data_types::f32, format::bfzyx
#define CASE_GATHER_ELEMENTS_FP32_5D_2 data_types::f32, format::bfzyx, { 5, 4, 7, 4, 4 }, format::bfzyx, { 5, 4, 7, 4, 3 }, format::bfzyx, { 5, 4, 7, 4, 3 }, 2, data_types::f32, format::bfzyx

#define CASE_GATHER_ELEMENTS_FP32_6D_1 data_types::f32, format::bfwzyx, { 5, 4, 6, 7, 8, 2 }, format::bfwzyx, { 5, 2, 6, 7, 8, 2 }, format::bfwzyx, { 5, 2, 6, 7, 8, 2 }, 1, data_types::f32, format::bfwzyx
#define CASE_GATHER_ELEMENTS_FP32_6D_2 data_types::f32, format::bfwzyx, { 2, 1, 2, 3, 2, 1 }, format::bfwzyx, { 2, 1, 2, 3, 2, 3 }, format::bfwzyx, { 2, 1, 2, 3, 2, 3 }, 2, data_types::f32, format::bfwzyx
#define CASE_GATHER_ELEMENTS_FP32_6D_3 data_types::f32, format::bfwzyx, { 2, 2, 3, 4, 4, 2 }, format::bfwzyx, { 2, 2, 6, 4, 4, 2 }, format::bfwzyx, { 2, 2, 6, 4, 4, 2 }, 5, data_types::f32, format::bfwzyx

class gather_elements_quantize : public GatherElementsPrimitiveFusingTest {};
TEST_P(gather_elements_quantize, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("gather_elements_indices", get_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p))-1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gather_elements("gather_elements_prim", input_info("input"), input_info("gather_elements_indices"), p.output_format, p.output_shape, p.axis),
        quantize("quantize", input_info("gather_elements_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_elements_quantize, ::testing::ValuesIn(std::vector<gather_elements_test_params>{
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_4D_1, 2, 3 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_4D_2, 2, 3 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_4D_3, 2, 3 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_5D_1, 2, 3 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_5D_2, 2, 3 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_6D_1, 2, 3 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_6D_2, 2, 3 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_6D_3, 2, 3 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_4D_1, 2, 3 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_4D_2, 2, 3 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_4D_3, 2, 3 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_5D_1, 2, 3 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_5D_2, 2, 3 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_6D_1, 2, 3 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_6D_2, 2, 3 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_6D_3, 2, 3 },
}));


class gather_elements_scale_activation : public GatherElementsPrimitiveFusingTest {};
TEST_P(gather_elements_scale_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("gather_elements_indices", get_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p))-1)),
        data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
        gather_elements("gather_elements_prim", input_info("input"), input_info("gather_elements_indices"), p.output_format, p.output_shape, p.axis),
        activation("activation", input_info("gather_elements_prim"), activation_func::abs),
        eltwise("scale", { input_info("activation"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_elements_scale_activation, ::testing::ValuesIn(std::vector<gather_elements_test_params>{
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_4D_1, 2, 4 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_4D_2, 2, 4 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_4D_3, 2, 4 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_5D_1, 2, 4 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_5D_2, 2, 4 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_6D_1, 2, 4 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_6D_2, 2, 4 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_6D_3, 2, 4 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_4D_1, 2, 4 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_4D_2, 2, 4 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_4D_3, 2, 4 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_5D_1, 2, 4 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_5D_2, 2, 4 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_6D_1, 2, 4 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_6D_2, 2, 4 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_6D_3, 2, 4 },
}));


class gather_elements_activation_scale_eltwise : public GatherElementsPrimitiveFusingTest {};
TEST_P(gather_elements_activation_scale_eltwise, basic) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("gather_elements_indices", get_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p))-1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 255)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        gather_elements("gather_elements_prim", input_info("input"), input_info("gather_elements_indices"), p.output_format, p.output_shape, p.axis),
        activation("activation", input_info("gather_elements_prim"), activation_func::abs),
        eltwise("scale", { input_info("activation"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        eltwise("eltwise", { input_info("scale"), input_info("eltwise_data") }, eltwise_mode::sum, p.data_type),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_elements_activation_scale_eltwise, ::testing::ValuesIn(std::vector<gather_elements_test_params>{
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_4D_1, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_4D_2, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_4D_3, 2, 5 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_5D_1, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_5D_2, 2, 5 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_6D_1, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_6D_2, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP16_6D_3, 2, 5 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_4D_1, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_4D_2, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_4D_3, 2, 5 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_5D_1, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_5D_2, 2, 5 },

    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_6D_1, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_6D_2, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ELEMENTS_FP32_6D_3, 2, 5 },
}));
