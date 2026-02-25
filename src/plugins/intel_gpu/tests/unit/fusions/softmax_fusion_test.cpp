// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/activation.hpp>
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
    int64_t dimension;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    std::string kernel_name;
};

class SoftmaxPrimitiveFusingTest : public ::BaseFusingTest<softmax_test_params> {
public:

    void execute(softmax_test_params& p, std::map<std::string, std::vector<std::string>> expected_fused_primitives_ids = {}) {
        auto input_prim = get_mem(get_input_layout(p));

        if (!p.kernel_name.empty()) {
            ov::intel_gpu::ImplementationDesc softmax_target_impl = { format::bfyx, p.kernel_name };
            cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"softmax", softmax_target_impl} }));
        }

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
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
#define CASE_SOFTMAX_FP32_1 {1, 15, 4, 5}, data_types::f32, format::bfyx, 1, data_types::f32, format::bfyx
#define CASE_SOFTMAX_FP32_2 {1, 15, 4, 5}, data_types::f32, format::bfyx, 3, data_types::f32, format::bfyx
#define CASE_SOFTMAX_FP32_3 {1, 15, 1, 1}, data_types::f32, format::bfyx, 1, data_types::f32, format::bfyx
#define CASE_SOFTMAX_FP32_4 {1, 15, 1, 2}, data_types::f32, format::bfyx, 2, data_types::f32, format::bfyx

#define CASE_SOFTMAX_FP16_1 {1, 15, 4, 5}, data_types::f16, format::bfyx, 1, data_types::f16, format::bfyx
#define CASE_SOFTMAX_FP16_2 {1, 15, 4, 5}, data_types::f16, format::bfyx, 3, data_types::f16, format::bfyx
#define CASE_SOFTMAX_FP16_3 {1, 15, 1, 1}, data_types::f16, format::bfyx, 1, data_types::f16, format::bfyx
#define CASE_SOFTMAX_FP16_4 {1, 15, 1, 2}, data_types::f16, format::bfyx, 2, data_types::f16, format::bfyx

class softmax_quantize : public SoftmaxPrimitiveFusingTest {};
TEST_P(softmax_quantize, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        softmax("softmax", input_info("input"), p.dimension),
        quantize("quantize", input_info("softmax"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), get_output_layout(p))
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, softmax_quantize,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_SOFTMAX_FP32_1, 2, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP32_2, 3, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP32_3, 2, 3 },

                        softmax_test_params{ CASE_SOFTMAX_FP16_1, 2, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP16_2, 3, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP16_3, 2, 3 },
}));

class softmax_activation : public SoftmaxPrimitiveFusingTest {};
TEST_P(softmax_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        softmax("softmax", input_info("input"), p.dimension),
        activation("log", input_info("softmax"), activation_func::log),
        reorder("reorder_bfyx", input_info("log"), get_output_layout(p))
    );

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, softmax_activation,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_SOFTMAX_FP32_3, 2, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_FP32_4, 2, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_FP16_3, 2, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_FP16_4, 2, 3, "softmax_gpu_bf" }
}));

class softmax_quantize_fusing_through : public SoftmaxPrimitiveFusingTest {};
TEST_P(softmax_quantize_fusing_through, reshape) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        softmax("softmax", input_info("input"), p.dimension),
        reshape("reshape", input_info("softmax"), get_reshape_shape(p)),
        quantize("quantize", input_info("reshape"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), get_output_layout(p))
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
        softmax("softmax", input_info("input"), p.dimension),
        reorder("reorder", input_info("softmax"), reorder_layout),
        quantize("quantize", input_info("reorder"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), get_output_layout(p))
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
        softmax("softmax", input_info("input"), p.dimension),
        reshape("reshape_first", input_info("softmax"), get_reshape_shape(p)),
        reorder("reorder", input_info("reshape_first"), reorder_layout),
        reshape("reshape_second", input_info("reorder"), get_reshape_shape(p)),
        quantize("quantize", input_info("reshape_second"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), get_output_layout(p))
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
