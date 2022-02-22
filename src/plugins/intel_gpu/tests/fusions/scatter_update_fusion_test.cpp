// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/scatter_update.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct scatter_update_test_params {
    tensor dictionary_shape;
    tensor indices_shape;
    tensor updates_shape;
    cldnn::scatter_update::scatter_update_axis axis;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class ScatterUpdatePrimitiveFusingTest : public ::BaseFusingTest<scatter_update_test_params> {
public:
    void execute(scatter_update_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(scatter_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.dictionary_shape };
    }

    layout get_indices_layout(scatter_update_test_params& p) {
        return layout{ p.data_type, format::bfyx, p.indices_shape };
    }

    layout get_updates_layout(scatter_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.updates_shape };
    }

    size_t get_axis_dim(scatter_update_test_params& p) {
        switch (p.axis) {
            case cldnn::scatter_update::scatter_update_axis::along_x:
                return p.dictionary_shape.spatial[0];
            case cldnn::scatter_update::scatter_update_axis::along_y:
                return p.dictionary_shape.spatial[1];
            case cldnn::scatter_update::scatter_update_axis::along_z:
                return p.dictionary_shape.spatial[2];
            case cldnn::scatter_update::scatter_update_axis::along_w:
                return p.dictionary_shape.spatial[3];
            case cldnn::scatter_update::scatter_update_axis::along_f:
                return p.dictionary_shape.feature[0];
            case cldnn::scatter_update::scatter_update_axis::along_b:
                return p.dictionary_shape.batch[0];
            default:
                return 1;
        }
    }

    layout get_per_channel_layout(scatter_update_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.dictionary_shape.feature[0], 1, 1 } };
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ ScatterUpdate cases -------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_SCATTER_UPDATE_FP32_1 { 2, 4, 1, 1 }, { 2, 1, 1, 1 }, { 2, 4, 1, 1 }, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_UPDATE_FP32_2 { 8, 1, 1, 1 }, { 4, 1, 1, 1 }, { 4, 1, 1, 1 }, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_UPDATE_FP32_3 { 4, 3, 1, 1 }, { 2, 2, 1, 1 }, { 2, 2, 1, 3 }, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_UPDATE_FP32_4 { 2, 5, 1, 2 }, { 2, 2, 1, 1 }, { 2, 2, 2, 2 }, cldnn::scatter_update::scatter_update_axis::along_f, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_UPDATE_FP32_5 { 2, 2, 1, 4 }, { 2, 2, 1, 1 }, { 2, 2, 2, 2 }, cldnn::scatter_update::scatter_update_axis::along_y, data_types::f32, format::bfyx, data_types::f32, format::bfyx

#define CASE_SCATTER_UPDATE_FP16_1 { 2, 4, 1, 1 }, { 1, 1, 1, 2 }, { 2, 1, 2, 1 }, cldnn::scatter_update::scatter_update_axis::along_f, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_UPDATE_FP16_2 { 8, 2, 1, 20 }, { 2, 3, 1, 1 }, { 2, 3, 20, 2 }, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_UPDATE_FP16_3 { 2, 2, 4, 1 }, { 3, 1, 1, 1 }, { 2, 2, 3, 1 }, cldnn::scatter_update::scatter_update_axis::along_x, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_UPDATE_FP16_4 { 6, 2, 1, 1 }, { 1, 2, 1, 2 }, { 1, 2, 2, 2 }, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_UPDATE_FP16_5 { 3, 1, 1, 5 }, { 2, 2, 1, 1 }, { 3, 1, 2, 2 }, cldnn::scatter_update::scatter_update_axis::along_y, data_types::f16, format::bfyx, data_types::f16, format::bfyx

#define CASE_SCATTER_UPDATE_5D_FP32_1 { 4, 3, 1, 4, 1 }, { 4, 1, 1, 1 }, { 4, 3, 1, 4, 1 }, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP32_2 { 2, 3, 2, 2, 2 }, { 2, 1, 1, 1 }, { 2, 2, 2, 2, 2 }, cldnn::scatter_update::scatter_update_axis::along_f, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP32_3 { 5, 3, 2, 4, 2 }, { 3, 1, 1, 1 }, { 5, 3, 2, 3, 2 }, cldnn::scatter_update::scatter_update_axis::along_y, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP32_4 { 2, 3, 1, 4, 4 }, { 2, 1, 1, 1 }, { 2, 3, 1, 4, 2 }, cldnn::scatter_update::scatter_update_axis::along_z, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP32_5 { 3, 1, 5, 2, 1 }, { 2, 1, 1, 1 }, { 3, 1, 2, 2, 1 }, cldnn::scatter_update::scatter_update_axis::along_x, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_SCATTER_UPDATE_5D_FP16_1 { 3, 2, 1, 2, 1 }, { 2, 1, 1, 1 }, { 2, 2, 2, 2, 1 }, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP16_2 { 1, 3, 1, 2, 1 }, { 2, 1, 1, 1 }, { 1, 2, 1, 2, 1 }, cldnn::scatter_update::scatter_update_axis::along_f, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP16_3 { 2, 3, 1, 3, 3 }, { 1, 2, 1, 1 }, { 2, 3, 1, 2, 3 }, cldnn::scatter_update::scatter_update_axis::along_y, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP16_4 { 3, 2, 2, 2, 2 }, { 2, 1, 1, 1 }, { 3, 2, 2, 2, 2 }, cldnn::scatter_update::scatter_update_axis::along_z, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP16_5 { 1, 1, 4, 1, 1 }, { 3, 1, 1, 1 }, { 1, 1, 3, 1, 1 }, cldnn::scatter_update::scatter_update_axis::along_x, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx

class scatter_update_quantize : public ScatterUpdatePrimitiveFusingTest {};
TEST_P(scatter_update_quantize, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scatter_update_indices", get_repeatless_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)) - 1)),
        data("scatter_update_updates", get_mem(get_updates_layout(p), 0, 1000)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        scatter_update("scatter_update_prim", "input", "scatter_update_indices", "scatter_update_updates", p.axis),
        quantize("quantize", "scatter_update_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_update_quantize, ::testing::ValuesIn(std::vector<scatter_update_test_params>{
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_1, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_2, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_3, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_4, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_5, 2, 3 },

    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_1, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_2, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_3, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_4, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_5, 2, 3 },

    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_1, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_2, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_3, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_4, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_5, 2, 3 },

    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_1, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_2, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_3, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_4, 2, 3 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_5, 2, 3 },
}));

class scatter_update_scale_activation : public ScatterUpdatePrimitiveFusingTest {};
TEST_P(scatter_update_scale_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scatter_update_indices", get_repeatless_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)) - 1)),
        data("scatter_update_updates", get_mem(get_updates_layout(p), 0, 1000)),
        data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
        scatter_update("scatter_update_prim", "input", "scatter_update_indices", "scatter_update_updates", p.axis),
        activation("activation", "scatter_update_prim", activation_func::abs),
        scale("scale", "activation", "scale_data"),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_update_scale_activation, ::testing::ValuesIn(std::vector<scatter_update_test_params>{
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_1, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_2, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_3, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_4, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_5, 2, 4 },

    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_1, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_2, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_3, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_4, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_5, 2, 4 },

    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_1, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_2, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_3, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_4, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_5, 2, 4 },

    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_1, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_2, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_3, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_4, 2, 4 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_5, 2, 4 },
}));

class scatter_update_scale_activation_eltwise : public ScatterUpdatePrimitiveFusingTest {};
TEST_P(scatter_update_scale_activation_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scatter_update_indices", get_repeatless_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)) - 1)),
        data("scatter_update_updates", get_mem(get_updates_layout(p), 0, 1000)),
        data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
        data("eltw_data", get_mem(layout(p.default_type, p.default_format, p.dictionary_shape))),
        scatter_update("scatter_update_prim", "input", "scatter_update_indices", "scatter_update_updates", p.axis),
        activation("activation", "scatter_update_prim", activation_func::abs),
        eltwise("eltw", { "activation", "eltw_data" }, eltwise_mode::sum, p.default_type),
        scale("scale", "eltw", "scale_data"),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_update_scale_activation_eltwise, ::testing::ValuesIn(std::vector<scatter_update_test_params>{
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_1, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_2, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_3, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_4, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_5, 3, 5 },

    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_1, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_2, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_3, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_4, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_5, 3, 5 },

    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_1, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_2, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_3, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_4, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_5, 3, 5 },

    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_1, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_2, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_3, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_4, 3, 5 },
    scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_5, 3, 5 },
}));
