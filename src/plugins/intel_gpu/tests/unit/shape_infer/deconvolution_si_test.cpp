// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "deconvolution_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct deconvolution_test_params {
    ov::PartialShape input_shape;
    ov::PartialShape weight_shape;
    uint32_t groups;
    ov::Strides stride;
    ov::Strides dilations;
    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    ov::CoordinateDiff output_padding;
    bool with_output_shape;
    ov::PartialShape output_pshape;
    layout expected_layout;
    std::vector<int64_t> output_shape_data;
    bool with_bias;
};

class deconvolution_si_test : public testing::TestWithParam<deconvolution_test_params> { };

TEST_P(deconvolution_si_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();
    auto input_data_layout = layout{p.input_shape, data_types::f32, format::bfyx};
    auto weight_layout = layout{p.weight_shape, data_types::f32, format::bfyx};

    cldnn::primitive_id weights = "weight";
    cldnn::primitive_id bias = "";

    auto input_prim = std::make_shared<input_layout>("data", input_data_layout);
    auto weight_prim = std::make_shared<input_layout>("weight", weight_layout);

    std::shared_ptr<data> bias_prim;
    if (p.with_bias) {
        bias = "bias";
        const int64_t out_features = static_cast<int64_t>(p.expected_layout.get_partial_shape()[1].get_length());
        auto bias_layout = layout{ov::PartialShape{1, out_features, 1, 1}, data_types::f32, format::bfyx};
        auto bias_mem = engine.allocate_memory(bias_layout);
        set_values<float>(bias_mem, std::vector<float>(bias_layout.count(), 1.0f));
        bias_prim = std::make_shared<data>(bias, bias_mem);
    }

    auto deconv_prim = std::make_shared<deconvolution>("deconv", input_info("data"), weights, bias, p.groups,
                                                       p.stride, p.pads_begin, p.dilations, p.pads_begin,
                                                       p.pads_end, p.output_padding, false);
    if (p.with_output_shape) {
        deconv_prim->output_partial_shape = p.output_pshape;
    }

    std::shared_ptr<data> out_shape_prim;
    if (!p.output_shape_data.empty()) {
        auto out_shape_layout = layout{ov::PartialShape{static_cast<int64_t>(p.output_shape_data.size())},
                                       data_types::i64, format::bfyx};
        auto out_shape_mem = engine.allocate_memory(out_shape_layout);
        set_values<int64_t>(out_shape_mem, p.output_shape_data);
        out_shape_prim = std::make_shared<data>("output_shape", out_shape_mem);
        deconv_prim->output_shape_id = input_info("output_shape");
    }

    cldnn::program prog(engine);

    auto& input_node = prog.get_or_create(input_prim);
    auto& weight_node = prog.get_or_create(weight_prim);
    auto& deconv_node = prog.get_or_create(deconv_prim);
    program_wrapper::add_connection(prog, input_node, deconv_node);
    program_wrapper::add_connection(prog, weight_node, deconv_node);
    if (p.with_bias) {
        auto& bias_node = prog.get_or_create(bias_prim);
        program_wrapper::add_connection(prog, bias_node, deconv_node);
    }
    if (!p.output_shape_data.empty()) {
        auto& out_shape_node = prog.get_or_create(out_shape_prim);
        program_wrapper::add_connection(prog, out_shape_node, deconv_node);

        auto shape_infer_deps = deconv_node.get_shape_infer_dependencies();
        ASSERT_EQ(shape_infer_deps.size(), 1u);
        ASSERT_EQ(shape_infer_deps.front(), p.with_bias ? 3u : 2u);
    }

    auto params = deconv_node.get_kernel_impl_params();
    auto res = deconvolution_inst::calc_output_layouts<ov::PartialShape>(deconv_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, deconvolution_si_test,
    testing::ValuesIn(std::vector<deconvolution_test_params>{
        // 2d deconv
        {
            ov::PartialShape{1, 20, 224, 224}, ov::PartialShape{10, 20, 3, 3},
            1, {2, 2}, {1, 1},
            std::vector<ptrdiff_t>{1, 1}, std::vector<ptrdiff_t>{1, 1},
            std::vector<ptrdiff_t>{0, 0},
            false, {},
            layout{ov::PartialShape{1, 10, 447, 447}, data_types::f32, format::bfyx}
        },
        // 2d deconv with output padding
        {
            ov::PartialShape{1, 20, 2, 2}, ov::PartialShape{10, 20, 3, 3},
            1, {3, 3}, {1, 1},
            std::vector<ptrdiff_t>{0, 0}, std::vector<ptrdiff_t>{0, 0},
            std::vector<ptrdiff_t>{2, 2},
            false, {},
            layout{ov::PartialShape{1, 10, 8, 8}, data_types::f32, format::bfyx}
        },
        // 2d deconv with dynamic shape
        {
            ov::PartialShape::dynamic(4), ov::PartialShape{10, 20, 3, 3},
            1, {3, 3}, {1, 1},
            std::vector<ptrdiff_t>{0, 0}, std::vector<ptrdiff_t>{0, 0},
            std::vector<ptrdiff_t>{2, 2},
            false, {},
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}
        },
        // 1d groupdeconv
        {
            ov::PartialShape{1, 20, 224}, ov::PartialShape{4, 2, 5, 3},
            4, {2}, {1},
            std::vector<ptrdiff_t>{1}, std::vector<ptrdiff_t>{1},
            std::vector<ptrdiff_t>{0},
            false, {},
            layout{ov::PartialShape{1, 8, 447}, data_types::f32, format::bfyx}
        },
        // 2d groupdeconv
        {
            ov::PartialShape{1, 20, 224, 224}, ov::PartialShape{4, 2, 5, 3, 3},
            4, {2, 2}, {1, 1},
            std::vector<ptrdiff_t>{1, 1}, std::vector<ptrdiff_t>{1, 1},
            std::vector<ptrdiff_t>{0, 0},
            false, {},
            layout{ov::PartialShape{1, 8, 447, 447}, data_types::f32, format::bfyx}
        },
    }));

INSTANTIATE_TEST_SUITE_P(smoke_with_output_shape, deconvolution_si_test,
    testing::ValuesIn(std::vector<deconvolution_test_params>{
        // 2d deconv with output shape
        {
            ov::PartialShape{1, 20, 224, 224}, ov::PartialShape{10, 20, 3, 3},
            1, {2, 2}, {1, 1},
            std::vector<ptrdiff_t>{1, 1}, std::vector<ptrdiff_t>{1, 1},
            std::vector<ptrdiff_t>{0, 0},
            true, ov::PartialShape{500, 500},
            layout{ov::PartialShape{1, 10, 500, 500}, data_types::f32, format::bfyx}
        },
        // 1d groupdeconv with output shape
        {
            ov::PartialShape{1, 20, 224}, ov::PartialShape{4, 2, 5, 3},
            4, {2}, {1},
            std::vector<ptrdiff_t>{1}, std::vector<ptrdiff_t>{1},
            std::vector<ptrdiff_t>{0},
            true, ov::PartialShape{500},
            layout{ov::PartialShape{1, 8, 500}, data_types::f32, format::bfyx}
        },
        // 2d groupdeconv with output shape
        {
            ov::PartialShape{1, 20, 224, 224}, ov::PartialShape{4, 2, 5, 3, 3},
            4, {2, 2}, {1, 1},
            std::vector<ptrdiff_t>{1, 1}, std::vector<ptrdiff_t>{1, 1},
            std::vector<ptrdiff_t>{0, 0},
            true, ov::PartialShape{500, 500},
            layout{ov::PartialShape{1, 8, 500, 500}, data_types::f32, format::bfyx}
        },
    }));

INSTANTIATE_TEST_SUITE_P(smoke_with_output_shape_from_data, deconvolution_si_test,
    testing::ValuesIn(std::vector<deconvolution_test_params>{
        // 2d deconv, runtime output shape, no bias (output shape dep at index 2)
        {
            ov::PartialShape{1, 20, 224, 224}, ov::PartialShape{10, 20, 3, 3},
            1, {2, 2}, {1, 1},
            std::vector<ptrdiff_t>{1, 1}, std::vector<ptrdiff_t>{1, 1},
            std::vector<ptrdiff_t>{0, 0},
            false, {},
            layout{ov::PartialShape{1, 10, 500, 500}, data_types::f32, format::bfyx},
            std::vector<int64_t>{500, 500}, false
        },
        // 2d deconv, runtime output shape, with fused bias (output shape dep at index 3)
        {
            ov::PartialShape{1, 20, 224, 224}, ov::PartialShape{10, 20, 3, 3},
            1, {2, 2}, {1, 1},
            std::vector<ptrdiff_t>{1, 1}, std::vector<ptrdiff_t>{1, 1},
            std::vector<ptrdiff_t>{0, 0},
            false, {},
            layout{ov::PartialShape{1, 10, 500, 500}, data_types::f32, format::bfyx},
            std::vector<int64_t>{500, 500}, true
        },
        // 2d groupdeconv, runtime output shape, with fused bias (output shape dep at index 3)
        {
            ov::PartialShape{1, 20, 224, 224}, ov::PartialShape{4, 2, 5, 3, 3},
            4, {2, 2}, {1, 1},
            std::vector<ptrdiff_t>{1, 1}, std::vector<ptrdiff_t>{1, 1},
            std::vector<ptrdiff_t>{0, 0},
            false, {},
            layout{ov::PartialShape{1, 8, 500, 500}, data_types::f32, format::bfyx},
            std::vector<int64_t>{500, 500}, true
        },
    }));

}  // shape_infer_tests
