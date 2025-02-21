// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations/common_optimizations/moc_transformations.hpp"
#include "transformations/common_optimizations/ric_fusion.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;
using namespace opset8;

namespace {
std::shared_ptr<Convolution> create_conv(Output<Node> input, Output<Node> weights) {
    return std::make_shared<Convolution>(input,
                                         weights,
                                         ov::Strides{1, 1},
                                         ov::CoordinateDiff{0, 0},
                                         ov::CoordinateDiff{0, 0},
                                         ov::Strides{1, 1});
}

std::shared_ptr<Constant> create_weights(const Shape& weigts_shape) {
    std::vector<float> values(ov::shape_size(weigts_shape));
    float cur_value = 0.01f;
    for (auto& value : values) {
        value = cur_value;
        cur_value += 0.01f;
    }
    return Constant::create(element::f32, weigts_shape, values);
}

std::shared_ptr<Convolution> create_conv(Output<Node> input, const Shape& weigts_shape) {
    return create_conv(input, create_weights(weigts_shape));
}

std::shared_ptr<GroupConvolution> create_group_conv(Output<Node> input, Output<Node> weights) {
    return std::make_shared<GroupConvolution>(input,
                                              weights,
                                              ov::Strides{1, 1},
                                              ov::CoordinateDiff{0, 0},
                                              ov::CoordinateDiff{0, 0},
                                              ov::Strides{1, 1});
}

std::shared_ptr<GroupConvolution> create_group_conv(Output<Node> input, const Shape& weigts_shape) {
    return create_group_conv(input, create_weights(weigts_shape));
}

std::shared_ptr<GroupConvolution> create_group_conv_with_gather(Output<Node> input,
                                                                const Shape& weigts_shape,
                                                                const std::vector<int64_t>& order) {
    auto gather = std::make_shared<Gather>(create_weights(weigts_shape),
                                           Constant::create(element::i64, Shape{order.size()}, order),
                                           Constant::create(element::i64, Shape{1}, {0}));
    return std::make_shared<GroupConvolution>(input,
                                              ov::util::get_constant_from_source(gather),
                                              ov::Strides{1, 1},
                                              ov::CoordinateDiff{0, 0},
                                              ov::CoordinateDiff{0, 0},
                                              ov::Strides{1, 1});
}

std::shared_ptr<Convolution> create_conv_with_gather(Output<Node> input,
                                                     Output<Node> weigts,
                                                     const std::vector<int64_t>& order) {
    auto gather = std::make_shared<Gather>(weigts,
                                           Constant::create(element::i64, Shape{order.size()}, order),
                                           Constant::create(element::i64, Shape{1}, {1}));
    return std::make_shared<Convolution>(input,
                                         ov::util::get_constant_from_source(gather),
                                         ov::Strides{1, 1},
                                         ov::CoordinateDiff{0, 0},
                                         ov::CoordinateDiff{0, 0},
                                         ov::Strides{1, 1});
}

std::shared_ptr<Convolution> create_conv_with_gather(Output<Node> input,
                                                     const Shape& weigts_shape,
                                                     const std::vector<int64_t>& order) {
    return create_conv_with_gather(input, create_weights(weigts_shape), order);
}

std::shared_ptr<Parameter> create_param(const PartialShape& shape) {
    return std::make_shared<Parameter>(element::f32, shape);
}

std::shared_ptr<Gather> create_gather(Output<Node> input, std::vector<int64_t> order, int64_t axis) {
    auto order_const = Constant::create(element::i64, Shape{order.size()}, order);
    auto axis_const = Constant::create(element::i64, Shape{1}, {axis});
    return std::make_shared<Gather>(input, order_const, axis_const);
}

std::shared_ptr<FakeQuantize> create_fq(Output<Node> input) {
    return std::make_shared<FakeQuantize>(input,
                                          Constant::create(element::f32, Shape{1}, {1}),
                                          Constant::create(element::f32, Shape{1}, {2}),
                                          Constant::create(element::f32, Shape{1}, {3}),
                                          Constant::create(element::f32, Shape{1}, {4}),
                                          255);
}

void apply_reverse_input_channels(std::shared_ptr<Model> f, std::vector<std::pair<int64_t, std::string>> data) {
    using namespace ov::preprocess;
    PrePostProcessor p(f);
    for (auto item : data) {
        p.input(item.first).tensor().set_layout(ov::Layout(item.second));
        p.input(item.first).preprocess().reverse_channels();
    }
    p.build();
}

}  // namespace

TEST_F(TransformationTestsF, RICFusionSimple) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto relu = std::make_shared<Relu>(input);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto relu = std::make_shared<Relu>(input);
        auto conv = create_conv_with_gather(relu, {6, 3, 3, 3}, {2, 1, 0});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionHard) {
    {
        auto input = create_param({-1, -1, -1, -1});
        auto relu = std::make_shared<Relu>(input);

        auto input2 = create_param({-1, 3, -1, -1});
        auto split = std::make_shared<Split>(input2, Constant::create(element::i64, {}, {1}), 3);
        auto concat = std::make_shared<Concat>(OutputVector{split->output(2), split->output(1), split->output(0)}, 1);
        auto eltwise = std::make_shared<Add>(relu, concat);
        auto prelu = std::make_shared<PRelu>(eltwise, Constant::create(element::f32, Shape{}, {6.0f}));

        auto gconv = create_group_conv(prelu, {3, 4, 1, 3, 3});

        auto pow = std::make_shared<Power>(gconv, Constant::create(element::f32, Shape{}, {-1.0f}));
        auto convert1 = std::make_shared<Convert>(pow, element::f16);
        auto convert2 = std::make_shared<Convert>(convert1, element::f32);

        auto gconv2 = create_group_conv(convert2, {12, 1, 1, 3, 3});

        auto conv = create_conv(gconv2, {6, 12, 3, 3});
        auto conv2 = create_conv(concat, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv, conv2}, ParameterVector{input, input2});

        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
    {
        auto input = create_param({-1, -1, -1, -1});
        auto relu = std::make_shared<Relu>(input);

        auto input2 = create_param({-1, 3, -1, -1});
        auto eltwise = std::make_shared<Add>(relu, input2);
        auto prelu = std::make_shared<PRelu>(eltwise, Constant::create(element::f32, Shape{}, {6.0f}));
        //       0            1            2                 2              1            0
        // [0, 1, 2, 3]-[4, 5, 6, 7]-[8, 9, 10, 11] -> [8, 9, 10, 11]-[4, 5, 6, 7]-[0, 1, 2, 3]
        auto gconv = create_group_conv_with_gather(prelu, {3, 4, 1, 3, 3}, {2, 1, 0});

        auto pow = std::make_shared<Power>(gconv, Constant::create(element::f32, Shape{}, {-1.0f}));
        auto convert1 = std::make_shared<Convert>(pow, element::f16);
        auto convert2 = std::make_shared<Convert>(convert1, element::f32);

        auto gconv2 = create_group_conv_with_gather(convert2, {12, 1, 1, 3, 3}, {8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3});

        auto conv = create_conv_with_gather(gconv2, {6, 12, 3, 3}, {8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3});
        auto conv2 = create_conv_with_gather(input2, {6, 3, 3, 3}, {2, 1, 0});

        model_ref = std::make_shared<Model>(NodeVector{conv, conv2}, ParameterVector{input, input2});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionHardNegativePad12) {
    {
        auto input = create_param({-1, -1, -1, -1});
        auto relu = std::make_shared<Relu>(input);

        auto input2 = create_param({-1, 3, -1, -1});
        auto split = std::make_shared<Split>(input2, Constant::create(element::i64, {}, {1}), 3);
        auto concat = std::make_shared<Concat>(OutputVector{split->output(2), split->output(1), split->output(0)}, 1);
        auto eltwise = std::make_shared<Add>(relu, concat);

        auto pads_begin = Constant::create(element::i64, Shape{4}, {0, 0, 0, -1});
        auto pads_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, -1});
        auto pad = std::make_shared<ov::op::v12::Pad>(eltwise, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto gconv = create_group_conv(pad, {3, 4, 1, 3, 3});

        auto pow = std::make_shared<Power>(gconv, Constant::create(element::f32, Shape{}, {-1.0f}));
        auto convert1 = std::make_shared<Convert>(pow, element::f16);
        auto convert2 = std::make_shared<Convert>(convert1, element::f32);

        auto gconv2 = create_group_conv(convert2, {12, 1, 1, 3, 3});

        auto conv = create_conv(gconv2, {6, 12, 3, 3});
        auto conv2 = create_conv(concat, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv, conv2}, ParameterVector{input, input2});

        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
    {
        auto input = create_param({-1, -1, -1, -1});
        auto relu = std::make_shared<Relu>(input);

        auto input2 = create_param({-1, 3, -1, -1});
        auto eltwise = std::make_shared<Add>(relu, input2);

        auto pads_begin = Constant::create(element::i64, Shape{4}, {0, 0, 0, -1});
        auto pads_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, -1});
        auto pad = std::make_shared<ov::op::v12::Pad>(eltwise, pads_begin, pads_end, op::PadMode::CONSTANT);

        //       0            1            2                 2              1            0
        // [0, 1, 2, 3]-[4, 5, 6, 7]-[8, 9, 10, 11] -> [8, 9, 10, 11]-[4, 5, 6, 7]-[0, 1, 2, 3]
        auto gconv = create_group_conv_with_gather(pad, {3, 4, 1, 3, 3}, {2, 1, 0});

        auto pow = std::make_shared<Power>(gconv, Constant::create(element::f32, Shape{}, {-1.0f}));
        auto convert1 = std::make_shared<Convert>(pow, element::f16);
        auto convert2 = std::make_shared<Convert>(convert1, element::f32);

        auto gconv2 = create_group_conv_with_gather(convert2, {12, 1, 1, 3, 3}, {8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3});

        auto conv = create_conv_with_gather(gconv2, {6, 12, 3, 3}, {8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3});
        auto conv2 = create_conv_with_gather(input2, {6, 3, 3, 3}, {2, 1, 0});

        model_ref = std::make_shared<Model>(NodeVector{conv, conv2}, ParameterVector{input, input2});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionDynamic) {
    {
        auto input = create_param({-1, -1, -1, -1});
        auto relu = std::make_shared<Relu>(input);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }

    {
        auto input = create_param({-1, -1, -1, -1});
        auto relu = std::make_shared<Relu>(input);
        auto conv = create_conv_with_gather(relu, {6, 3, 3, 3}, {2, 1, 0});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, RICFusionEltwise1) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto add = std::make_shared<Add>(input, Constant::create(element::f32, Shape{3, 1, 1}, {0.1, 0.2, 0.3}));
        auto conv = create_conv(add, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto gather = create_gather(Constant::create(element::f32, Shape{3, 1, 1}, {0.1, 0.2, 0.3}), {2, 1, 0}, 0);
        auto add = std::make_shared<Add>(input, ov::util::get_constant_from_source(gather));
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwise2) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto add = std::make_shared<Add>(input, create_weights({1, 1, 1}));
        auto conv = create_conv(add, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto add = std::make_shared<Add>(input, create_weights({1, 1, 1}));
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwise3) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto add = std::make_shared<Add>(input, Constant::create(element::f32, Shape{1}, {0.2}));
        auto conv = create_conv(add, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto add = std::make_shared<Add>(input, Constant::create(element::f32, Shape{1}, {0.2}));
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwise4) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto add = std::make_shared<Add>(create_weights({3, 1, 1}), input);
        auto conv = create_conv(add, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto gather = create_gather(create_weights({3, 1, 1}), {2, 1, 0}, 0);
        auto add = std::make_shared<Add>(ov::util::get_constant_from_source(gather), input);
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwise5) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto add = std::make_shared<Add>(create_weights({1, 3, 1, 1}), input);
        auto conv = create_conv(add, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto gather = create_gather(create_weights({1, 3, 1, 1}), {2, 1, 0}, 1);
        auto add = std::make_shared<Add>(ov::util::get_constant_from_source(gather), input);
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwiseNegative) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto input2 = create_param({1, 3, 64, 64});
        auto add = std::make_shared<Add>(input, input2);
        auto conv = create_conv(add, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input, input2});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
}

TEST_F(TransformationTestsF, RICFusionEltwiseTwoRIC) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto input2 = create_param({1, 1, 64, 64});
        auto add = std::make_shared<Add>(input, input2);
        auto conv = create_conv(add, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input, input2});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}, {1, "NCHW"}});
    }
    {
        auto input = create_param({1, 3, 64, 64});
        auto input2 = create_param({1, 1, 64, 64});
        auto add = std::make_shared<Add>(input, input2);
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input, input2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwiseNegative3) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto add = std::make_shared<Add>(input, Constant::create(element::f32, {1, 1, 1, 1, 1}, {1.4}));
        auto shapeof = std::make_shared<ShapeOf>(add);

        model = std::make_shared<Model>(NodeVector{shapeof}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
}

TEST_F(TransformationTestsF, RICFusionGroupConv) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto gconv = create_group_conv(input, {3, 2, 1, 3, 3});
        auto relu = std::make_shared<Relu>(gconv);
        auto conv = create_conv(relu, {3, 6, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
    {
        auto input = create_param({1, 3, 64, 64});
        auto gconv = create_group_conv_with_gather(input, {3, 2, 1, 3, 3}, {2, 1, 0});
        //    0     1      2          2     1      0
        // [0, 1]-[2, 3]-[4, 5] -> [4, 5]-[2, 3]-[0, 1]
        auto relu = std::make_shared<Relu>(gconv);
        auto conv = create_conv_with_gather(relu, {3, 6, 3, 3}, {4, 5, 2, 3, 0, 1});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionGroupConvNegative) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto gconv = create_group_conv(input, {1, 3, 3, 3, 3});
        auto relu = std::make_shared<Relu>(gconv);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
}

TEST_F(TransformationTestsF, RICFusionTranspose) {
    {
        auto input = create_param({1, 64, 64, 3});
        auto add = std::make_shared<Add>(input, create_weights({3}));
        auto transpose = std::make_shared<Transpose>(add, Constant::create(element::i64, Shape{4}, {0, 3, 1, 2}));
        auto conv = create_conv(transpose, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NHWC"}});
    }

    {
        auto input = create_param({1, 64, 64, 3});
        auto gather = create_gather(create_weights({3}), {2, 1, 0}, 0);
        auto add = std::make_shared<Add>(input, ov::util::get_constant_from_source(gather));
        auto transpose = std::make_shared<Transpose>(add, Constant::create(element::i64, Shape{4}, {0, 3, 1, 2}));
        auto conv = create_conv_with_gather(transpose, {6, 3, 3, 3}, {2, 1, 0});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionFQOnTheWay) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto fq = create_fq(input);
        auto conv = create_conv(fq, create_fq(create_weights({6, 3, 3, 3})));

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto fq = create_fq(input);
        auto weights = ov::util::get_constant_from_source(create_gather(create_weights({6, 3, 3, 3}), {2, 1, 0}, 1));
        auto conv = create_conv(fq, create_fq(weights));

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionFQOnTheWay2) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto fq = create_fq(input);
        auto weights_const = create_weights({6, 3, 3, 3});
        auto fq_weights = std::make_shared<FakeQuantize>(weights_const,
                                                         create_weights({1, 3, 1, 1}),
                                                         create_weights({1, 1, 1}),
                                                         create_weights({1}),
                                                         create_weights({3, 1, 1}),
                                                         255);
        auto conv = create_conv(fq, fq_weights);

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto fq = create_fq(input);
        auto weights_const = create_weights({6, 3, 3, 3});
        auto fq_weights = std::make_shared<FakeQuantize>(
            ov::util::get_constant_from_source(create_gather(weights_const, {2, 1, 0}, 1)),
            ov::util::get_constant_from_source(create_gather(create_weights({1, 3, 1, 1}), {2, 1, 0}, 1)),
            create_weights({1, 1, 1}),
            create_weights({1}),
            ov::util::get_constant_from_source(create_gather(create_weights({3, 1, 1}), {2, 1, 0}, 0)),
            255);
        auto conv = create_conv(fq, fq_weights);

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionFQOnTheWay3) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto fq = create_fq(input);
        auto weights_const = create_weights({3, 1, 1, 3, 3});
        auto fq_weights = std::make_shared<FakeQuantize>(weights_const,
                                                         create_weights({3, 1, 1, 1, 1}),
                                                         create_weights({1, 1, 1}),
                                                         create_weights({1}),
                                                         create_weights({1}),
                                                         255);
        auto gconv = create_group_conv(fq, fq_weights);
        auto conv = create_conv(gconv, {6, 3, 1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto fq = create_fq(input);
        auto weights_const = create_weights({3, 1, 1, 3, 3});
        auto fq_weights = std::make_shared<FakeQuantize>(
            ov::util::get_constant_from_source(create_gather(weights_const, {2, 1, 0}, 0)),
            ov::util::get_constant_from_source(create_gather(create_weights({3, 1, 1, 1, 1}), {2, 1, 0}, 0)),
            create_weights({1, 1, 1}),
            create_weights({1}),
            create_weights({1}),
            255);
        auto gconv = create_group_conv(fq, fq_weights);
        auto conv = create_conv_with_gather(gconv, {6, 3, 1, 1}, {2, 1, 0});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionShapeOf) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto relu = std::make_shared<Relu>(input);
        auto shape_of = std::make_shared<ShapeOf>(relu);

        model = std::make_shared<Model>(NodeVector{shape_of}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto relu = std::make_shared<Relu>(input);
        auto shape_of = std::make_shared<ShapeOf>(relu);

        model_ref = std::make_shared<Model>(NodeVector{shape_of}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, RICFusionGatherDetectionNegative) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto gather = create_gather(input, {2, 1, 1}, 1);
        auto relu = std::make_shared<Relu>(gather);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        manager.register_pass<ov::pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionGatherDetectionNegative2) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto input2 = std::make_shared<Parameter>(element::i64, Shape{3});
        auto gather = std::make_shared<Gather>(input, input2, Constant::create(element::i64, {}, {1}));
        auto relu = std::make_shared<Relu>(gather);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input, input2});
        manager.register_pass<ov::pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionGatherDetectionNegative3) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto gather = create_gather(input, {1}, 1);
        auto relu = std::make_shared<Relu>(gather);
        auto conv = create_conv(relu, {6, 1, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        manager.register_pass<ov::pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionGatherDetectionNegative4) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto gather = create_gather(input, {2, 0}, 1);
        auto relu = std::make_shared<Relu>(gather);
        auto conv = create_conv(relu, {6, 2, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        manager.register_pass<ov::pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionSplitConcatDetectionNegative) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto split = std::make_shared<Split>(input, Constant::create(element::i64, {}, {1}), 3);
        auto concat = std::make_shared<Concat>(OutputVector{split->output(2), split->output(1), split->output(1)}, 1);
        auto relu = std::make_shared<Relu>(concat);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        manager.register_pass<ov::pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionSplitConcatDetectionNegative2) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto split = std::make_shared<Split>(input, Constant::create(element::i64, {}, {1}), 3);
        auto split2 = std::make_shared<Split>(input, Constant::create(element::i64, {}, {1}), 3);
        auto concat = std::make_shared<Concat>(OutputVector{split->output(2), split->output(1), split2->output(0)}, 1);
        auto relu = std::make_shared<Relu>(concat);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        manager.register_pass<ov::pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionSplitConcatDetectionNegative3) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto split = std::make_shared<Split>(input, Constant::create(element::i64, {}, {1}), 3);
        auto concat = std::make_shared<Concat>(OutputVector{split->output(2), split->output(1), split->output(0)}, 1);
        auto relu = std::make_shared<Relu>(concat);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        model = std::make_shared<Model>(OutputVector{conv, split->output(0)}, ParameterVector{input});
        manager.register_pass<ov::pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, FuseConvertLayout) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 64});
        auto order = ov::op::v0::Constant::create(element::i64, Shape{3}, {1, 2, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input, order);
        auto relu = std::make_shared<ov::op::v0::Relu>(transpose);

        model = std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{input});

        using namespace ov::preprocess;
        PrePostProcessor p(model);
        p.input(0).tensor().set_element_type(element::f16);
        p.input(0).preprocess().convert_layout({2, 0, 1});
        p.build();

        manager.register_pass<ov::pass::TransposeSinking>();
    }

    {
        auto input = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{3, 64, 1});
        auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
        auto relu = std::make_shared<ov::op::v0::Relu>(convert);

        model_ref = std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, FuseScaleValue) {
    {
        auto input = create_param({1, 64, 64, 3});
        auto order = ov::op::v0::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input, order);
        auto conv = create_conv(transpose, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});

        using namespace ov::preprocess;
        PrePostProcessor p(model);
        p.input(0).tensor().set_layout("NHWC");
        p.input(0).preprocess().scale(1.3f);
        p.build();

        manager.register_pass<ov::pass::MOCTransformations>(false);
    }

    {
        auto input = create_param({1, 64, 64, 3});
        auto order = ov::op::v0::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input, order);
        auto conv = create_conv(transpose, {6, 3, 3, 3});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, FuseScaleValues) {
    {
        auto input = create_param({1, 64, 64, 3});
        auto order = ov::op::v0::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input, order);
        auto conv = create_conv(transpose, {6, 3, 3, 3});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});

        using namespace ov::preprocess;
        PrePostProcessor p(model);
        p.input(0).tensor().set_layout("NHWC");
        p.input(0).preprocess().scale({1.3f, 0.2f, 4.1f});
        p.build();

        manager.register_pass<ov::pass::MOCTransformations>(false);
    }

    {
        auto input = create_param({1, 64, 64, 3});
        auto order = ov::op::v0::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input, order);
        auto conv = create_conv(transpose, {6, 3, 3, 3});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionConvertMultiply) {
    // Input graph:
    //
    //     Parameter
    //      |F32
    //      |
    //     FakeQuantize
    //      |F32
    //      |
    //     Convert             Constant
    //      |U8                   |I8
    //      |                     |
    //     Convert  Constant   Convert(DCF) Constant
    //       \FP32    /FP32       \FP32    /F32
    //        \      /             \      /
    //         Multiply            Multiply
    //             \FP32            /FP32
    //              \              /
    //                 Convolution
    //
    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset8::FakeQuantize>(parameter,
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {20}),
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {254}),
                                                   255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{4, 3, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset8::Convert>(weights, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(convert, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset8::Convolution>(activations,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        model = std::make_shared<ov::Model>(conv, ParameterVector{parameter});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset8::FakeQuantize>(parameter,
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {20}),
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {254}),
                                                   255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{4, 3, 1, 1}, {-2});
        {
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto gather = ov::util::get_constant_from_source(create_gather(weights, {2, 1, 0}, 1));
            auto convert = std::make_shared<opset8::Convert>(gather, element::f32);
            auto multiply = std::make_shared<opset8::Multiply>(convert, scale);
            weights = multiply;
        }
        auto conv = std::make_shared<opset8::Convolution>(activations,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(conv, ParameterVector{parameter});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionConvertMultiplyGroupConv) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        std::shared_ptr<Node> weights = opset8::Constant::create(element::f32, Shape{3, 3, 1, 4, 4}, {-2});
        auto convert = std::make_shared<opset8::Convert>(weights, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
        auto multiply = std::make_shared<opset8::Multiply>(convert, scale);

        auto group_conv = std::make_shared<opset8::GroupConvolution>(data,
                                                                     multiply,
                                                                     Strides{1, 1},
                                                                     CoordinateDiff{1, 1},
                                                                     CoordinateDiff{3, 3},
                                                                     Shape{1, 1},
                                                                     op::PadType::EXPLICIT);
        auto relu = std::make_shared<Relu>(group_conv);
        auto conv = create_conv(relu, {6, 9, 3, 3});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        std::shared_ptr<Node> weights = opset8::Constant::create(element::f32, Shape{3, 3, 1, 4, 4}, {-2});
        auto gather = create_gather(weights, {2, 1, 0}, 1);
        auto convert = std::make_shared<opset8::Convert>(gather, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
        auto multiply = ov::util::get_constant_from_source(std::make_shared<opset8::Multiply>(convert, scale));

        auto group_conv = std::make_shared<opset8::GroupConvolution>(data,
                                                                     multiply,
                                                                     Strides{1, 1},
                                                                     CoordinateDiff{1, 1},
                                                                     CoordinateDiff{3, 3},
                                                                     Shape{1, 1},
                                                                     op::PadType::EXPLICIT);
        auto relu = std::make_shared<Relu>(group_conv);
        std::shared_ptr<Node> weights2 = opset8::Constant::create(element::f32, Shape{6, 9, 3, 3}, {-2});
        auto gather2 = ov::util::get_constant_from_source(create_gather(weights2, {6, 7, 8, 3, 4, 5, 0, 1, 2}, 1));
        auto conv = std::make_shared<opset8::Convolution>(relu,
                                                          gather2,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::Strides{1, 1});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionConvertMultiplyNegative1) {
    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset8::FakeQuantize>(parameter,
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {20}),
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {254}),
                                                   255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{4, 3, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset8::Convert>(weights, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{1, 1, 1, 1}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(convert, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset8::Convolution>(activations,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        model = std::make_shared<ov::Model>(conv, ParameterVector{parameter});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset8::FakeQuantize>(parameter,
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {20}),
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {254}),
                                                   255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{4, 3, 1, 1}, {-2});
        {
            auto gather = ov::util::get_constant_from_source(create_gather(weights, {2, 1, 0}, 1));
            auto convert = std::make_shared<opset8::Convert>(gather, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{1, 1, 1, 1}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(convert, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset8::Convolution>(activations,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(conv, ParameterVector{parameter});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionConvertMultiplyNegativeBroadcast) {
    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset8::FakeQuantize>(parameter,
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {20}),
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {254}),
                                                   255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{3, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset8::Convert>(weights, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{4, 3, 1, 1}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(convert, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset8::Convolution>(activations,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        model = std::make_shared<ov::Model>(conv, ParameterVector{parameter});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        std::shared_ptr<Node> activations =
            std::make_shared<opset8::FakeQuantize>(parameter,
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {20}),
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {254}),
                                                   255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{3, 1, 1}, {-2});
        {
            auto gather = ov::util::get_constant_from_source(create_gather(weights, {2, 1, 0}, 0));
            auto convert = std::make_shared<opset8::Convert>(gather, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{4, 3, 1, 1}, {0.2});
            auto gather2 = ov::util::get_constant_from_source(create_gather(scale, {2, 1, 0}, 1));
            auto multiply = std::make_shared<opset8::Multiply>(convert, gather2);
            weights = multiply;
        }

        auto conv = std::make_shared<opset8::Convolution>(activations,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(conv, ParameterVector{parameter});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionNegativeUnsupported) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto relu = std::make_shared<Relu>(input);
        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{6, 3, 3, 3}, {-2});
        {
            auto convert = std::make_shared<opset8::Convert>(weights, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(convert, scale);
            auto relu2 = std::make_shared<Relu>(multiply);
            weights = relu2;
        }
        auto conv = std::make_shared<opset8::Convolution>(relu,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
        apply_reverse_input_channels(model, {{0, "NCHW"}});
        manager.register_pass<ov::pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionConvertMultiplyNonScalarFQInput) {
    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        std::shared_ptr<Node> activations = std::make_shared<opset8::FakeQuantize>(
            parameter,
            std::make_shared<opset8::Constant>(element::f32, Shape{1, 3, 14, 14}),
            opset8::Constant::create(element::f32, Shape{}, {20}),
            opset8::Constant::create(element::f32, Shape{}, {0}),
            opset8::Constant::create(element::f32, Shape{}, {254}),
            255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{4, 3, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset8::Convert>(weights, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(convert, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset8::Convolution>(activations,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        model = std::make_shared<ov::Model>(conv, ParameterVector{parameter});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        auto gather =
            create_gather(std::make_shared<opset8::Constant>(element::f32, Shape{1, 3, 14, 14}), {2, 1, 0}, 1);
        std::shared_ptr<Node> activations =
            std::make_shared<opset8::FakeQuantize>(parameter,
                                                   ov::util::get_constant_from_source(gather),
                                                   opset8::Constant::create(element::f32, Shape{}, {20}),
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {254}),
                                                   255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{4, 3, 1, 1}, {-2});
        {
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto gather = ov::util::get_constant_from_source(create_gather(weights, {2, 1, 0}, 1));
            auto convert = std::make_shared<opset8::Convert>(gather, element::f32);
            auto multiply = std::make_shared<opset8::Multiply>(convert, scale);
            weights = multiply;
        }
        auto conv = std::make_shared<opset8::Convolution>(activations,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(conv, ParameterVector{parameter});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionConvertMultiplySkipIfFQLowNonConst) {
    {
        auto parameter = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        auto input_low = std::make_shared<opset8::Parameter>(element::f32, Shape{});
        std::shared_ptr<Node> activations =
            std::make_shared<opset8::FakeQuantize>(parameter,
                                                   input_low,
                                                   opset8::Constant::create(element::f32, Shape{}, {20}),
                                                   opset8::Constant::create(element::f32, Shape{}, {0}),
                                                   opset8::Constant::create(element::f32, Shape{}, {254}),
                                                   255);
        {
            auto first_convert = std::make_shared<opset8::Convert>(activations, element::u8);
            auto second_convert = std::make_shared<opset8::Convert>(first_convert, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(second_convert, scale);
            activations = multiply;
        }

        std::shared_ptr<Node> weights = opset8::Constant::create(element::i8, Shape{4, 3, 1, 1}, {-2});
        {
            auto convert = std::make_shared<opset8::Convert>(weights, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {0.2});
            auto multiply = std::make_shared<opset8::Multiply>(convert, scale);
            weights = multiply;
        }

        auto conv = std::make_shared<opset8::Convolution>(activations,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        model = std::make_shared<ov::Model>(conv, ParameterVector{parameter, input_low});
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
    manager.register_pass<ov::pass::ReverseInputChannelsFusion>();
}

TEST_F(TransformationTestsF, RICFusionTwoConvolutions) {
    auto input = create_param({1, 3, 16, 16});
    {
        auto conv1 = create_conv(input, create_weights({3, 3, 1, 1}));
        auto conv2 = create_conv(conv1, create_weights({3, 3, 1, 1}));
        model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
    {
        auto conv1_with_gather = create_conv_with_gather(input, create_weights({3, 3, 1, 1}), {2, 1, 0});
        auto conv2 = create_conv(conv1_with_gather, create_weights({3, 3, 1, 1}));
        model_ref = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionTwoConvolutionsTheSameWeights) {
    auto input = create_param({1, 3, 16, 16});
    auto weights = create_weights({3, 3, 1, 1});
    {
        auto conv1 = create_conv(input, weights);
        auto conv2 = create_conv(conv1, weights);
        model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});

        // ReverseInputChannelsFusion is expected to be applied inside PrePostProcessing
        apply_reverse_input_channels(model, {{0, "NCHW"}});
    }
    {
        auto conv1_with_gather = create_conv_with_gather(input, weights, {2, 1, 0});
        auto conv2 = create_conv(conv1_with_gather, weights);
        model_ref = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
