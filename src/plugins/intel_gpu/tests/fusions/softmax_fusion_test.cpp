// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct softmax_test_params {
    tensor in_shape;
    data_types data_type;
    format input_format;
    softmax::dimension_t dimension;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class SoftmaxPrimitiveFusingTest : public ::BaseFusingTest<softmax_test_params> {
public:

    void execute(softmax_test_params& p, std::map<std::string, std::vector<std::string>> expected_fused_primitives_ids = {}) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        check_fusions_correctness(network_fused, expected_fused_primitives_ids);
    }

    layout get_input_layout(softmax_test_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape, padding{} };
    }

    tensor get_reshape_shape(softmax_test_params& p) {
        auto output_shape = p.in_shape;
        output_shape.feature[0] *= output_shape.spatial[0];
        output_shape.spatial[0] = 1;
        return output_shape;
    }

    layout get_output_layout(softmax_test_params& p) {
        return { data_types::f32, p.default_format, get_reshape_shape(p) };
    }
};

}  // namespace


/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- SoftMax cases ---------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
#define CASE_SOFTMAX_FP32_1 {1, 15, 4, 5}, data_types::f32, format::bfyx, softmax::dimension_t::normalize_f, data_types::f32, format::bfyx
#define CASE_SOFTMAX_FP32_2 {1, 15, 4, 5}, data_types::f32, format::bfyx, softmax::dimension_t::normalize_x, data_types::f32, format::bfyx
#define CASE_SOFTMAX_FP32_3 {1, 15, 4, 5}, data_types::f32, format::bfyx, softmax::dimension_t::normalize_fyx, data_types::f32, format::bfyx

#define CASE_SOFTMAX_FP16_1 {1, 15, 4, 5}, data_types::f16, format::bfyx, softmax::dimension_t::normalize_f, data_types::f16, format::bfyx
#define CASE_SOFTMAX_FP16_2 {1, 15, 4, 5}, data_types::f16, format::bfyx, softmax::dimension_t::normalize_x, data_types::f16, format::bfyx
#define CASE_SOFTMAX_FP16_3 {1, 15, 4, 5}, data_types::f16, format::bfyx, softmax::dimension_t::normalize_fyx, data_types::f16, format::bfyx

class softmax_quantize : public SoftmaxPrimitiveFusingTest {};
TEST_P(softmax_quantize, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        softmax("softmax", "input", p.dimension),
        quantize("quantize", "softmax", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", get_output_layout(p))
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, softmax_quantize,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_SOFTMAX_FP32_1, 2, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP32_2, 3, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP32_3, 3, 3 },

                        softmax_test_params{ CASE_SOFTMAX_FP16_1, 2, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP16_2, 3, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP16_3, 3, 3 },
}));

class softmax_quantize_fusing_through : public SoftmaxPrimitiveFusingTest {};
TEST_P(softmax_quantize_fusing_through, reshape) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        softmax("softmax", "input", p.dimension),
        reshape("reshape", "softmax", get_reshape_shape(p)),
        quantize("quantize", "reshape", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", get_output_layout(p))
    );

    tolerance = 1.f;
    execute(p, {{"softmax", {"quantize"}}});
}

TEST_P(softmax_quantize_fusing_through, reorder) {
    auto p = GetParam();
    auto reorder_layout = layout{ p.data_type, p.input_format, get_reshape_shape(p), padding{} };
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        softmax("softmax", "input", p.dimension),
        reorder("reorder", "softmax", reorder_layout),
        quantize("quantize", "reorder", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", get_output_layout(p))
    );

    tolerance = 1.f;
    execute(p, {{"softmax", {"quantize"}}});
}

TEST_P(softmax_quantize_fusing_through, chain) {
    auto p = GetParam();
    auto reorder_layout = layout{ p.data_type, p.input_format, get_reshape_shape(p), padding{} };
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        softmax("softmax", "input", p.dimension),
        reshape("reshape_first", "softmax", get_reshape_shape(p)),
        reorder("reorder", "reshape_first", reorder_layout),
        reshape("reshape_second", "reorder", get_reshape_shape(p)),
        quantize("quantize", "reshape_second", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", get_output_layout(p))
    );

    tolerance = 1.f;
    execute(p, {{"softmax", {"quantize"}}});
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, softmax_quantize_fusing_through,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_SOFTMAX_FP32_1, 2, 3 },
                        // Such fusions not allowed yet
                        // softmax_test_params{ CASE_SOFTMAX_FP32_2, 3, 3 },
                        // softmax_test_params{ CASE_SOFTMAX_FP32_3, 3, 3 },

                        softmax_test_params{ CASE_SOFTMAX_FP16_1, 2, 3 },
                        // Such fusions not allowed yet
                        // softmax_test_params{ CASE_SOFTMAX_FP16_2, 3, 3 },
                        // softmax_test_params{ CASE_SOFTMAX_FP16_3, 3, 3 },
}));
