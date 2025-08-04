// Copyright (C) 2023 Intel Corporation
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
    auto deconv_prim = std::make_shared<deconvolution>("deconv", input_info("data"), weights, bias, p.groups,
                                                       p.stride, p.pads_begin, p.dilations, p.pads_begin,
                                                       p.pads_end, p.output_padding, false);
    if (p.with_output_shape) {
        deconv_prim->output_partial_shape = p.output_pshape;
    }

    cldnn::program prog(engine);

    auto& input_node = prog.get_or_create(input_prim);
    auto& weight_node = prog.get_or_create(weight_prim);
    auto& deconv_node = prog.get_or_create(deconv_prim);
    program_wrapper::add_connection(prog, input_node, deconv_node);
    program_wrapper::add_connection(prog, weight_node, deconv_node);

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

}  // shape_infer_tests
