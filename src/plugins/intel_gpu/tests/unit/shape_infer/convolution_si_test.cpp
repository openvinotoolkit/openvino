// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "convolution_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>
#include <optional>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

static void check_declared_output_type(std::optional<data_types> declared_type,
                                      data_types expected_type,
                                      bool fused = false) {
    auto& engine = get_test_engine();
    const layout input_layout{ov::PartialShape{1, 1, 1, 1}, data_types::f16, format::bfyx};
    auto input = std::make_shared<cldnn::input_layout>("input", input_layout);
    auto weights = std::make_shared<cldnn::input_layout>("weights", input_layout);
    auto convolution = std::make_shared<cldnn::convolution>(
        "conv", input_info("input"), "weights", "", 1, ov::Strides{1, 1}, ov::Strides{1, 1},
        ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, false);
    convolution->output_data_types = {declared_type};
    cldnn::program program(engine);
    auto& input_node = program.get_or_create(input);
    auto& weights_node = program.get_or_create(weights);
    auto& convolution_node = program.get_or_create(convolution);
    program_wrapper::add_connection(program, input_node, convolution_node);
    program_wrapper::add_connection(program, weights_node, convolution_node);
    const layout expected{ov::PartialShape{1, 1, 1, 1}, expected_type, format::bfyx};
    if (fused) {
        auto activation = std::make_shared<cldnn::activation>("activation", input_info("conv"), activation_func::abs);
        fused_primitive_desc fused_activation(activation);
        fused_activation.output_layouts = {expected};
        convolution_node.add_fused_primitive(fused_activation);
    }
    const auto params = convolution_node.get_kernel_impl_params();
    const auto outputs = convolution_inst::calc_output_layouts<ov::PartialShape>(convolution_node, *params);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0], expected);
    ASSERT_EQ(convolution_inst::calc_output_layout(convolution_node, *params), expected);
}

TEST(convolution_output_type, explicit_f32) {
    check_declared_output_type(data_types::f32, data_types::f32);
}

TEST(convolution_output_type, unspecified_preserves_input_type) {
    check_declared_output_type(std::nullopt, data_types::f16);
}

TEST(convolution_output_type, fused_output_takes_precedence) {
    check_declared_output_type(data_types::f32, data_types::f16, true);
}

struct convolution_test_params {
    ov::PartialShape input_shape;
    ov::PartialShape weight_shape;
    uint32_t groups;
    ov::Strides stride;
    ov::Strides dilation;
    ov::CoordinateDiff pad_above;
    ov::CoordinateDiff pad_below;
    layout expected_layout;
    bool weights_have_group_dim;
};

class convolution_si_test : public testing::TestWithParam<convolution_test_params> { };

TEST_P(convolution_si_test, shape_infer) {
    auto p = GetParam();
    auto& engine = get_test_engine();
    auto input_data_layout = layout{p.input_shape, data_types::f32, format::get_default_format(p.input_shape.size())};
    auto weight_layout = layout{p.weight_shape, data_types::f32, format::get_default_format(p.input_shape.size())};
    auto input_prim = std::make_shared<input_layout>("data", input_data_layout);
    auto weight_prim = std::make_shared<input_layout>("weight", weight_layout);
    std::shared_ptr<convolution> conv_prim = std::make_shared<convolution>("conv",
                                                                           input_info("data"),
                                                                           "weight",
                                                                           "",
                                                                           p.groups,
                                                                           p.stride,
                                                                           p.dilation,
                                                                           p.pad_above,
                                                                           p.pad_below,
                                                                           p.weights_have_group_dim);

    cldnn::program prog(engine);

    auto& input_node = prog.get_or_create(input_prim);
    auto& weight_node = prog.get_or_create(weight_prim);
    auto& conv_node = prog.get_or_create(conv_prim);
    program_wrapper::add_connection(prog, input_node, conv_node);
    program_wrapper::add_connection(prog, weight_node, conv_node);

    auto params = conv_node.get_kernel_impl_params();
    auto res = convolution_inst::calc_output_layouts<ov::PartialShape>(conv_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, convolution_si_test,
    testing::ValuesIn(std::vector<convolution_test_params>{
        // conv, symmetric pad
        {
            ov::PartialShape{1, 3, 10, 10}, ov::PartialShape{4, 3, 3, 3},          /* input, weight shape */
            1, {1, 1}, {1, 1},                                                     /* groups, stride, dilation */
            ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1},                    /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 10, 10}, data_types::f32, format::bfyx}, /* expected_layout */
            false                                                                  /* weight_has_groups */
        },
        // conv, symmetric pad
        {
            ov::PartialShape{1, 3, 10, 10}, ov::PartialShape{4, 3, 3, 3},          /* input, weight shape */
            1, {1, 1}, {1, 1},                                                     /* groups, stride, dilation */
            ov::CoordinateDiff{2, 1}, ov::CoordinateDiff{2, 1},                    /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 12, 10}, data_types::f32, format::bfyx}, /* expected_layout */
            false                                                                  /* weight_has_groups */
        },
        // conv, asymmetric pad
        {
            ov::PartialShape{1, 3, 10, 10}, ov::PartialShape{4, 3, 3, 3},          /* input, weight shape */
            1, {1, 1}, {1, 1},                                                     /* groups, stride, dilation */
            ov::CoordinateDiff{2, 1}, ov::CoordinateDiff{1, 2},                    /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 11, 11}, data_types::f32, format::bfyx}, /* expected_layout */
            false                                                                  /* weight_has_groups */
        },
        // conv, dynamic shape
        {
            ov::PartialShape::dynamic(4), ov::PartialShape{4, 3, 3, 3},                                  /* input, weight shape */
            1, {1, 1}, {1, 1},                                                                           /* groups, stride, dilation */
            ov::CoordinateDiff{1, 2}, ov::CoordinateDiff{2, 1},                                          /* pad_above, pad_below */
            layout{{-1, 4, ov::Dimension(1, -1), ov::Dimension(1, -1)}, data_types::f32, format::bfyx},  /* expected_layout */
            false                                                                                        /* weight_has_groups */
        },
        // groupconv, symmetric pad
        {
            ov::PartialShape{1, 12, 224, 224}, ov::PartialShape{4, 1, 3, 5, 5},       /* input, weight shape */
            4, {1, 1}, {1, 1},                                                        /* groups, stride, dilation */
            ov::CoordinateDiff{2, 2}, ov::CoordinateDiff{2, 2},                       /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 224, 224}, data_types::f32, format::bfyx},  /* expected_layout */
            true                                                                      /* weight_has_groups */
        },
        // groupconv, asymmetric pad
        {
            ov::PartialShape{1, 12, 224, 224}, ov::PartialShape{4, 1, 3, 5, 5},       /* input, weight shape */
            4, {1, 1}, {1, 1},                                                        /* groups, stride, dilation */
            ov::CoordinateDiff{2, 1}, ov::CoordinateDiff{1, 2},                       /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 223, 223}, data_types::f32, format::bfyx},  /* expected_layout */
            true                                                                      /* weight_has_groups */
        },
        // 5d grouped, asymmetric pad and kernel
        {
            ov::PartialShape{1, 12, 10, 20, 30}, ov::PartialShape{4, 1, 3, 3, 1, 3},    /* input, weight shape */
            4, {2, 1, 2}, {2, 2, 2},                                                    /* groups, stride, dilation */
            ov::CoordinateDiff{1, 0, 2}, ov::CoordinateDiff{1, 0, 0},                   /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 4, 20, 14}, data_types::f32, format::bfzyx},  /* expected_layout */
            true                                                                        /* weight_has_groups */
        },
        // 1d groupconv, asymmetric pad
        {
            ov::PartialShape{1, 12, 224}, ov::PartialShape{4, 1, 3, 5},               /* input, weight shape */
            4, {2}, {1},                                                              /* groups, stride, dilation */
            ov::CoordinateDiff{2}, ov::CoordinateDiff{1},                             /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 112}, data_types::f32, format::bfyx},       /* expected_layout */
            true                                                                      /* weight_has_groups */
        },
    }));
}
