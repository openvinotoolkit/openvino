// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gather.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct gather_test_params {
    ov::Shape dictionary_shape;
    ov::Shape indices_shape;
    ov::Shape out_shape;
    int64_t axis;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class GatherPrimitiveFusingTest : public ::BaseFusingTest<gather_test_params> {
public:
    void execute(gather_test_params& p, bool is_dynamic = false, bool count_reorder = false) {
        cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
        cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
        cfg_fused.set_property(ov::intel_gpu::optimize_data(true));
        auto input_prim = get_mem(get_input_layout(p));
        auto indices_prim = get_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p) - 1));
        auto elt_input_prim = get_mem(get_per_channel_layout(p), -10, 10);
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        if (is_dynamic) {
            network_fused.set_input_data("gather_indices", indices_prim);
            network_fused.set_input_data("eltwise_data", elt_input_prim);
            network_not_fused.set_input_data("gather_indices", indices_prim);
            network_not_fused.set_input_data("eltwise_data", elt_input_prim);
        }

        compare(network_not_fused, network_fused, p, count_reorder);
    }

    layout get_input_layout(gather_test_params& p, bool is_dynamic = false) {
        if (is_dynamic) {
            return layout{ ov::PartialShape::dynamic(p.dictionary_shape.size()), p.data_type, p.input_format };
        } else {
            return layout{ ov::PartialShape(p.dictionary_shape), p.data_type, p.input_format };
        }
    }

    layout get_indices_layout(gather_test_params& p) {
        return layout{ ov::PartialShape(p.indices_shape), p.data_type, format::bfyx };
    }

    size_t get_axis_dim(gather_test_params& p) {
        auto in_layout = get_input_layout(p);
        return in_layout.get_dims()[p.axis];
    }

    layout get_per_channel_layout(gather_test_params& p, bool is_dynamic = false) {
        std::vector<ov::Dimension> dims;
        for (size_t i = 0; i < p.out_shape.size(); ++i) {
            if (i != 1) {
                dims.push_back(ov::Dimension(1));
            } else {
                if (is_dynamic) {
                    dims.push_back(ov::Dimension(-1));
                } else {
                    dims.push_back(ov::Dimension(p.out_shape[1]));
                }
            }
        }
        return layout{ ov::PartialShape(dims), p.default_type, p.default_format };
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ Gather cases --------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_GATHER_FP32_1 { 2, 3, 4, 1 }, { 4, 1, 1, 1 }, { 4, 3, 4, 1 }, 0, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_2 { 3, 2, 2, 1 }, { 2, 3, 1, 1 }, { 2, 3, 2, 2 }, 0, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_3 { 3, 1, 2, 1 }, { 2, 1, 1, 1 }, { 3, 2, 2, 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_4 { 5, 3, 2, 2 }, { 3, 1, 1, 1 }, { 5, 2, 3, 2 }, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_5 { 2, 3, 2, 1 }, { 3, 1, 1, 1 }, { 2, 3, 3, 1 }, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_6 { 2, 3, 4 }, { 4 }, { 4, 3, 4 }, 0, data_types::f32, format::bfyx, data_types::f32, format::bfyx

#define CASE_GATHER_FP16_1 { 2, 3, 4, 1 }, { 4, 1, 1, 1 }, { 4, 3, 4, 1 }, 0, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_2 { 3, 2, 2, 1 }, { 2, 3, 1, 1 }, { 2, 3, 2, 2 }, 0, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_3 { 3, 1, 2, 1 }, { 2, 1, 1, 1 }, { 3, 2, 2, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_4 { 5, 2, 2, 2 }, { 3, 1, 1, 1 }, { 5, 2, 3, 2 }, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_5 { 2, 3, 2, 1 }, { 3, 1, 1, 1 }, { 2, 3, 3, 1 }, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_6 { 3, 2, 2 }, { 2, 3 }, { 2, 3, 2, 2 }, 0, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_7 { 2, 5, 2, 4 }, { 3, 2, 1}, { 2, 3, 2, 1, 2, 4 }, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx

#define CASE_GATHER_5D_FP32_1 { 2, 3, 1, 4, 1 }, { 4, 1, 1, 1 }, { 4, 3, 1, 4, 1 }, 0, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_2 { 2, 3, 2, 2, 2 }, { 2, 1, 1, 1 }, { 2, 2, 2, 2, 2 }, 1, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_3 { 5, 3, 2, 2, 2 }, { 3, 1, 1, 1 }, { 5, 3, 2, 3, 2 }, 3, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_4 { 2, 3, 1, 4, 4 }, { 2, 1, 1, 1 }, { 2, 3, 2, 4, 1 }, 2, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_5 { 3, 1, 5, 2, 1 }, { 2, 1, 1, 1 }, { 3, 1, 1, 2, 2 }, 4, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_GATHER_5D_FP16_1 { 3, 2, 1, 2, 1 }, { 2, 1, 1, 1 }, { 2, 2, 1, 2, 1 }, 0, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_2 { 1, 3, 1, 2, 1 }, { 2, 1, 1, 1 }, { 1, 2, 1, 2, 1 }, 1, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_3 { 2, 3, 3, 3, 1 }, { 1, 2, 1, 1 }, { 2, 3, 3, 1, 2 }, 3, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_4 { 3, 2, 2, 2, 2 }, { 2, 3, 1, 1 }, { 3, 2, 2, 3, 2 }, 2, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_5 { 1, 1, 2, 1, 1 }, { 3, 1, 1, 1 }, { 1, 1, 1, 1, 3 }, 4, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx

#define CASE_GATHER_INT8_1    { 2, 3, 4, 1 }, { 4 }, { 4, 3, 4, 1 }, 0, data_types::i8, format::bfyx, data_types::f32, format::bfyx

class gather_quantize : public GatherPrimitiveFusingTest {};
TEST_P(gather_quantize, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("gather_indices", get_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p) - 1))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gather("gather_prim", input_info("input"), input_info("gather_indices"), p.axis, p.dictionary_shape.size(), p.out_shape),
        quantize("quantize", input_info("gather_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_quantize, ::testing::ValuesIn(std::vector<gather_test_params>{
    gather_test_params{ CASE_GATHER_FP32_1, 2, 3 },
    gather_test_params{ CASE_GATHER_FP32_2, 2, 3 },
    gather_test_params{ CASE_GATHER_FP32_3, 2, 3 },
    gather_test_params{ CASE_GATHER_FP32_4, 2, 3 },
    gather_test_params{ CASE_GATHER_FP32_5, 2, 3 },

    gather_test_params{ CASE_GATHER_FP16_1, 2, 3 },
    gather_test_params{ CASE_GATHER_FP16_2, 2, 3 },
    gather_test_params{ CASE_GATHER_FP16_3, 2, 3 },
    gather_test_params{ CASE_GATHER_FP16_4, 2, 3 },
    gather_test_params{ CASE_GATHER_FP16_5, 2, 3 },

    gather_test_params{ CASE_GATHER_5D_FP32_1, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP32_2, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP32_3, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP32_4, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP32_5, 2, 3 },

    gather_test_params{ CASE_GATHER_5D_FP16_1, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP16_2, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP16_3, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP16_4, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP16_5, 2, 3 },
}));

class gather_eltwise_activation : public GatherPrimitiveFusingTest {};
TEST_P(gather_eltwise_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("gather_indices", get_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p) - 1))),
        data("eltwise_data", get_mem(get_per_channel_layout(p), -10, 10)),
        gather("gather_prim", input_info("input"), input_info("gather_indices"), p.axis, p.dictionary_shape.size(), p.out_shape),
        activation("activation", input_info("gather_prim"), activation_func::abs),
        eltwise("eltwise", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_eltwise_activation, ::testing::ValuesIn(std::vector<gather_test_params>{
    gather_test_params{ CASE_GATHER_FP32_1, 2, 4 },
    gather_test_params{ CASE_GATHER_FP32_2, 2, 4 },
    gather_test_params{ CASE_GATHER_FP32_3, 2, 4 },
    gather_test_params{ CASE_GATHER_FP32_4, 2, 4 },
    gather_test_params{ CASE_GATHER_FP32_5, 2, 4 },

    gather_test_params{ CASE_GATHER_FP16_1, 2, 4 },
    gather_test_params{ CASE_GATHER_FP16_2, 2, 4 },
    gather_test_params{ CASE_GATHER_FP16_3, 2, 4 },
    gather_test_params{ CASE_GATHER_FP16_4, 2, 4 },
    gather_test_params{ CASE_GATHER_FP16_5, 2, 4 },

    gather_test_params{ CASE_GATHER_5D_FP32_1, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP32_2, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP32_3, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP32_4, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP32_5, 2, 4 },

    gather_test_params{ CASE_GATHER_5D_FP16_1, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP16_2, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP16_3, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP16_4, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP16_5, 2, 4 },
}));

class gather_eltwise_activation_dynamic : public GatherPrimitiveFusingTest {};
TEST_P(gather_eltwise_activation_dynamic, basic) {
    auto p = GetParam();
    // Currently, eltwise fusion for dynamic shape + onednn is prevented
    // To be removed once dynamic shape fusion is allowed for onednn
    if (engine.get_device_info().supports_immad)
        return;

    create_topologies(
        input_layout("input", get_input_layout(p, true)),
        input_layout("gather_indices", layout{ ov::PartialShape::dynamic(p.indices_shape.size()), p.data_type, format::bfyx }),
        input_layout("eltwise_data", get_per_channel_layout(p, true)),
        gather("gather_prim", input_info("input"), input_info("gather_indices"), p.axis, p.dictionary_shape.size(), p.out_shape),
        activation("activation", input_info("gather_prim"), activation_func::abs),
        eltwise("eltwise", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32, std::vector<float>(), cldnn::reorder_mean_mode::subtract, cldnn::padding(), true)
    );

    tolerance = 1e-5f;
    execute(p, true, true);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_eltwise_activation_dynamic, ::testing::ValuesIn(std::vector<gather_test_params>{
    gather_test_params{ CASE_GATHER_FP32_6, 4, 6 },
    gather_test_params{ CASE_GATHER_FP16_6, 4, 7 },
    gather_test_params{ CASE_GATHER_FP16_7, 5, 8 },
    gather_test_params{ CASE_GATHER_INT8_1, 4, 7 },
}));
