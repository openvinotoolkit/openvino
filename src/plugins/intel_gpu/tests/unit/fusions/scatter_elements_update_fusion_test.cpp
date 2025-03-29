// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/scatter_elements_update.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct scatter_elements_update_test_params {
    tensor input_shape;
    tensor indices_shape;
    int64_t axis;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class ScatterElementsUpdatePrimitiveFusingTest : public ::BaseFusingTest<scatter_elements_update_test_params>{
public:
    void execute(scatter_elements_update_test_params& p) {

        auto input_prim = get_mem(get_input_layout(p), -5, 5);
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(scatter_elements_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.input_shape };
    }

    layout get_indices_layout(scatter_elements_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.indices_shape };
    }

    layout get_updates_layout(scatter_elements_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.indices_shape };
    }

    size_t get_axis_dim(scatter_elements_update_test_params& p) {
        auto shapes = p.input_shape.sizes(p.input_format);
        return shapes[p.axis];
    }

    layout get_per_channel_layout(scatter_elements_update_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.input_shape.feature[0], 1, 1 } };
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ ScatterElementsUpdate cases ------------------------------ */
/* ----------------------------------------------------------------------------------------------------- */

// input shape along the update axis should be larger than the total number of elements in the update tensor.
// This is not a limitation of operation itself, but a limitation of test implementation.
#define CASE_SCATTER_ELEMENTS_UPDATE_FP32_1 { 8, 4, 1, 1 }, { 2, 4, 1, 1 }, 0, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ELEMENTS_UPDATE_FP32_2 { 2, 8, 1, 2 }, { 2, 2, 1, 2 }, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ELEMENTS_UPDATE_FP32_3 { 2, 3, 10, 10 }, { 2, 2, 1, 2 }, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx

#define CASE_SCATTER_ELEMENTS_UPDATE_FP16_1 { 2, 2, 14, 12 }, { 2, 2, 3, 1 }, 3, data_types::f16, format::bfyx, data_types::f16, format::bfyx

#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_1 { 24, 3, 1, 4, 1 }, { 4, 3, 1, 2, 1 }, 0, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_2 { 2, 17, 2, 2, 2 }, { 1, 2, 2, 2, 2 }, 1, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_3 { 5, 3, 2, 20, 22 }, { 5, 1, 1, 2, 2 }, 3, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_1 { 13, 2, 1, 2, 1 }, { 2, 2, 1, 2, 1 }, 0, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_2 { 1, 13, 1, 2, 1 }, { 1, 2, 1, 2, 1 }, 1, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_3 { 2, 3, 1, 13, 13 }, { 2, 3, 1, 2, 1 }, 3, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx

class scatter_elements_update_quantize : public ScatterElementsUpdatePrimitiveFusingTest {};
TEST_P(scatter_elements_update_quantize, basic) {
    auto p = GetParam();
    const auto &seu = scatter_elements_update("scatter_elements_update_prim", input_info("input"), input_info("scatter_elements_update_indices"),
                                              input_info("scatter_elements_update_updates"), p.axis);
    const auto &q = quantize("quantize", input_info("scatter_elements_update_prim"), input_info("in_lo"), input_info("in_hi"),
                             input_info("out_lo"), input_info("out_hi"), 255, data_types::i8);
    const auto &r = reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32);
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scatter_elements_update_indices", get_repeatless_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)) - 1)),
        data("scatter_elements_update_updates", get_mem(get_updates_layout(p), 0, 100)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        seu,
        q,
        r
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_elements_update_quantize, ::testing::ValuesIn(std::vector<scatter_elements_update_test_params>{
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_1, 2, 3 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_2, 2, 3 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_3, 2, 3 },

    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP16_1, 2, 3 },

    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_1, 2, 3 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_2, 2, 3 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_3, 2, 3 },

    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_1, 2, 3 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_2, 2, 3 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_3, 2, 3 },
}));

class scatter_elements_update_scale_activation_eltwise : public ScatterElementsUpdatePrimitiveFusingTest {};
TEST_P(scatter_elements_update_scale_activation_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scatter_elements_update_indices", get_repeatless_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)) - 1)),
        data("scatter_elements_update_updates", get_mem(get_updates_layout(p), 0, 5)),
        data("scale_data", get_mem(get_per_channel_layout(p), -1, 1)),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.input_shape })),
        scatter_elements_update("scatter_elements_update_prim", input_info("input"), input_info("scatter_elements_update_indices"),
                                input_info("scatter_elements_update_updates"), p.axis),
        activation("activation", input_info("scatter_elements_update_prim"), activation_func::abs),
        eltwise("scale", { input_info("activation"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        eltwise("eltwise", { input_info("scale"), input_info("eltwise_data") }, eltwise_mode::sum, p.data_type),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_elements_update_scale_activation_eltwise, ::testing::ValuesIn(std::vector<scatter_elements_update_test_params>{
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_1, 2, 5 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_2, 2, 5 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_3, 2, 5 },

    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP16_1, 2, 5 },

    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_1, 2, 5 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_2, 2, 5 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_3, 2, 5 },

    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_1, 2, 5 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_2, 2, 5 },
    scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_3, 2, 5 },
}));
