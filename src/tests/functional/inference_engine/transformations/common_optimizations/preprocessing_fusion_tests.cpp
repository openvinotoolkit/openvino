// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/common_optimizations/ric_fusion.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/serialize.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;
using namespace opset8;

namespace {
std::shared_ptr<Convolution> create_conv(Output<Node> input, Output<Node> weights) {
    return std::make_shared<Convolution>(input, weights, ov::Strides{1, 1},
                                         ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                         ov::Strides{1, 1});
}

std::shared_ptr<Constant> create_weights(const Shape & weigts_shape) {
    std::vector<float> values(ov::shape_size(weigts_shape));
    float cur_value = 0.01;
    for (auto & value : values) {
        value = cur_value;
        cur_value += 0.01;
    }
    return Constant::create(element::f32, weigts_shape, values);
}

std::shared_ptr<Convolution> create_conv(Output<Node> input, const Shape & weigts_shape) {
    return create_conv(input, create_weights(weigts_shape));
}

std::shared_ptr<GroupConvolution> create_group_conv(Output<Node> input, Output<Node> weights) {
    return std::make_shared<GroupConvolution>(input, weights, ov::Strides{1, 1},
                                              ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                              ov::Strides{1, 1});
}

std::shared_ptr<GroupConvolution> create_group_conv(Output<Node> input, const Shape & weigts_shape) {
    return create_group_conv(input, create_weights(weigts_shape));
}

std::shared_ptr<GroupConvolution> create_group_conv_with_gather(Output<Node> input, const Shape & weigts_shape, const std::vector<int64_t> & order) {
    auto gather = std::make_shared<Gather>(create_weights(weigts_shape), Constant::create(element::i64, Shape{order.size()}, order),
                                           Constant::create(element::i64, Shape{1}, {0}));
    return std::make_shared<GroupConvolution>(input, gather, ov::Strides{1, 1},
                                              ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                              ov::Strides{1, 1});
}

std::shared_ptr<Convolution> create_conv_with_gather(Output<Node> input, const Shape & weigts_shape, const std::vector<int64_t> & order) {
    auto gather = std::make_shared<Gather>(create_weights(weigts_shape), Constant::create(element::i64, Shape{order.size()}, order),
                                                   Constant::create(element::i64, Shape{1}, {1}));
    return std::make_shared<Convolution>(input, gather, ov::Strides{1, 1},
                                                         ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                                         ov::Strides{1, 1});
}

std::shared_ptr<Parameter> create_param(const PartialShape & shape) {
    return std::make_shared<Parameter>(element::f32, shape);
}

std::shared_ptr<Gather> create_gather(Output<Node> input, std::vector<int64_t> order, int64_t axis) {
    auto order_const = Constant::create(element::i64, Shape{order.size()}, order);
    auto axis_const = Constant::create(element::i64, Shape{1}, {axis});
    return std::make_shared<Gather>(input, order_const, axis_const);
}

std::shared_ptr<FakeQuantize> create_fq(Output<Node> input) {
    return std::make_shared<FakeQuantize>(input, Constant::create(element::f32, Shape{1}, {1}),
                                                 Constant::create(element::f32, Shape{1}, {2}),
                                                 Constant::create(element::f32, Shape{1}, {3}),
                                                 Constant::create(element::f32, Shape{1}, {4}), 255);
}

void apply_reverse_input_channels(std::shared_ptr<Function> f, std::vector<std::pair<int64_t, std::string>> data) {
    using namespace ov::preprocess;
    PrePostProcessor p(f);
    for (auto item : data) {
        p.input(item.first).tensor().set_layout(ov::Layout(item.second));
        p.input(item.first).preprocess().reverse_channels();
    }
    p.build();
}

void apply_mean_value(std::shared_ptr<Function> f, std::vector<std::pair<int64_t, std::string>> data, std::vector<float> values) {
    using namespace ov::preprocess;
    PrePostProcessor p(f);
    for (auto item : data) {
        p.input(item.first).tensor().set_layout(ov::Layout(item.second));
        p.input(item.first).preprocess().mean(values);
    }
    p.build();
}

void apply_scale_value(std::shared_ptr<Function> f, std::vector<std::pair<int64_t, std::string>> data, std::vector<float> values) {
    using namespace ov::preprocess;
    PrePostProcessor p(f);
    for (auto item : data) {
        p.input(item.first).tensor().set_layout(ov::Layout(item.second));
        p.input(item.first).preprocess().scale(values);
    }
    p.build();
}
} // namespace

TEST_F(TransformationTestsF, RICFusionSimple) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto relu = std::make_shared<Relu>(input);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto relu = std::make_shared<Relu>(input);
        auto conv = create_conv_with_gather(relu, {6, 3, 3, 3}, {2, 1, 0});
        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionHard) {
    {
        auto input = create_param({ -1, -1, -1, -1 });
        auto relu = std::make_shared<Relu>(input);

        auto input2 = create_param({ -1, 3, -1, -1 });
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

        function = std::make_shared<Function>(NodeVector{ conv, conv2 }, ParameterVector{ input, input2 });

        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
    {
        auto input = create_param({ -1, -1, -1, -1 });
        auto relu = std::make_shared<Relu>(input);

        auto input2 = create_param({ -1, 3, -1, -1 });
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

        function_ref = std::make_shared<Function>(NodeVector{ conv, conv2 }, ParameterVector{ input, input2 });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionDynamic) {
    {
        auto input = create_param({ -1, -1, -1, -1 });
        auto relu = std::make_shared<Relu>(input);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({ -1, -1, -1, -1 });
        auto relu = std::make_shared<Relu>(input);
        auto conv = create_conv_with_gather(relu, {6, 3, 3, 3}, {2, 1, 0});
        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
}

TEST_F(TransformationTestsF, RICFusionEltwise1) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto add = std::make_shared<Add>(input, Constant::create(element::f32, Shape{3, 1, 1}, {0.1, 0.2, 0.3}));
        auto conv = create_conv(add, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto gather = create_gather(Constant::create(element::f32, Shape{3, 1, 1}, {0.1, 0.2, 0.3}), {2, 1, 0}, 0);
        auto add = std::make_shared<Add>(input, gather);
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});
        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwise2) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto add = std::make_shared<Add>(input, create_weights({1, 1, 1}));
        auto conv = create_conv(add, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto add = std::make_shared<Add>(input, create_weights({1, 1, 1}));
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});
        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwise3) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto add = std::make_shared<Add>(input, Constant::create(element::f32, Shape{1}, {0.2}));
        auto conv = create_conv(add, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto add = std::make_shared<Add>(input, Constant::create(element::f32, Shape{1}, {0.2}));
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});
        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwise4) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto add = std::make_shared<Add>(create_weights({3, 1, 1}), input);
        auto conv = create_conv(add, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto gather = create_gather(create_weights({3, 1, 1}), {2, 1, 0}, 0);
        auto add = std::make_shared<Add>(gather, input);
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});
        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwise5) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto add = std::make_shared<Add>(create_weights({1, 3, 1, 1}), input);
        auto conv = create_conv(add, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto gather = create_gather(create_weights({1, 3, 1, 1}), {2, 1, 0}, 1);
        auto add = std::make_shared<Add>(gather, input);
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});
        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwiseNegative) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto input2 = create_param({ 1, 3, 64, 64 });
        auto add = std::make_shared<Add>(input, input2);
        auto conv = create_conv(add, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input, input2 });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionEltwiseTwoRIC) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto input2 = create_param({ 1, 1, 64, 64 });
        auto add = std::make_shared<Add>(input, input2);
        auto conv = create_conv(add, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input, input2 });
        apply_reverse_input_channels(function, {{0, "NCHW"}, {1, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto input2 = create_param({ 1, 1, 64, 64 });
        auto add = std::make_shared<Add>(input, input2);
        auto conv = create_conv_with_gather(add, {6, 3, 3, 3}, {2, 1, 0});

        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input, input2 });
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionEltwiseNegative3) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto add = std::make_shared<Add>(input, Constant::create(element::f32, {1, 1, 1, 1, 1}, {1.4}));
        auto shapeof = std::make_shared<ShapeOf>(add);

        function = std::make_shared<Function>(NodeVector{ shapeof }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionGroupConv) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto gconv = create_group_conv(input, {3, 2, 1, 3, 3});
        auto relu = std::make_shared<Relu>(gconv);
        auto conv = create_conv(relu, {3, 6, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto gconv = create_group_conv_with_gather(input, {3, 2, 1, 3, 3}, {2, 1, 0});
        //    0     1      2          2     1      0
        // [0, 1]-[2, 3]-[4, 5] -> [4, 5]-[2, 3]-[0, 1]
        auto relu = std::make_shared<Relu>(gconv);
        auto conv = create_conv_with_gather(relu, {3, 6, 3, 3}, {4, 5, 2, 3, 0, 1});
        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionGroupConvNegative) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto gconv = create_group_conv(input, {1, 3, 3, 3, 3});
        auto relu = std::make_shared<Relu>(gconv);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionTranspose) {
    {
        auto input = create_param({ 1, 64, 64, 3 });
        auto add = std::make_shared<Add>(input, create_weights({3}));
        auto transpose = std::make_shared<Transpose>(add, Constant::create(element::i64, Shape{4}, {0, 3, 1, 2}));
        auto conv = create_conv(transpose, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NHWC"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({ 1, 64, 64, 3 });
        auto gather = create_gather(create_weights({3}), {2, 1, 0}, 0);
        auto add = std::make_shared<Add>(input, gather);
        auto transpose = std::make_shared<Transpose>(add, Constant::create(element::i64, Shape{4}, {0, 3, 1, 2}));
        auto conv = create_conv_with_gather(transpose, {6, 3, 3, 3}, {2, 1, 0});
        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionFQOnTheWay) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto fq = create_fq(input);
        auto conv = create_conv(fq, create_fq(create_weights({6, 3, 3, 3})));

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto fq = create_fq(input);
        auto conv = create_conv(fq, create_fq(create_gather(create_weights({6, 3, 3, 3}), {2, 1, 0}, 1)));

        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
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
                                                         create_weights({3, 1, 1}), 255);
        auto conv = create_conv(fq, fq_weights);

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto fq = create_fq(input);
        auto weights_const = create_weights({6, 3, 3, 3});
        auto fq_weights = std::make_shared<FakeQuantize>(create_gather(weights_const, {2, 1, 0}, 1),
                                                         create_gather(create_weights({1, 3, 1, 1}), {2, 1, 0}, 1),
                                                         create_weights({1, 1, 1}),
                                                         create_weights({1}),
                                                         create_gather(create_weights({3, 1, 1}), {2, 1, 0}, 0),
                                                         255);
        auto conv = create_conv(fq, fq_weights);

        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
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
                                                         create_weights({1}), 255);
        auto gconv = create_group_conv(fq, fq_weights);
        auto conv = create_conv(gconv, {6, 3, 1, 1});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto fq = create_fq(input);
        auto weights_const = create_weights({3, 1, 1, 3, 3});
        auto fq_weights = std::make_shared<FakeQuantize>(create_gather(weights_const, {2, 1, 0}, 0),
                                                         create_gather(create_weights({3, 1, 1, 1, 1}), {2, 1, 0}, 0),
                                                         create_weights({1, 1, 1}),
                                                         create_weights({1}),
                                                         create_weights({1}), 255);
        auto gconv = create_group_conv(fq, fq_weights);
        auto conv = create_conv_with_gather(gconv, {6, 3, 1, 1}, {2, 1, 0});

        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, RICFusionShapeOf) {
    {
        auto input = create_param({1, 3, 64, 64});
        auto relu = std::make_shared<Relu>(input);
        auto shape_of = std::make_shared<ShapeOf>(relu);

        function = std::make_shared<Function>(NodeVector{ shape_of }, ParameterVector{ input });
        apply_reverse_input_channels(function, {{0, "NCHW"}});

        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }

    {
        auto input = create_param({1, 3, 64, 64});
        auto relu = std::make_shared<Relu>(input);
        auto shape_of = std::make_shared<ShapeOf>(relu);

        function_ref = std::make_shared<Function>(NodeVector{ shape_of }, ParameterVector{ input });
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    disable_rt_info_check();
}

TEST_F(TransformationTestsF, RICFusionGatherDetectionNegative) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto gather = create_gather(input, {2, 1, 1}, 1);
        auto relu = std::make_shared<Relu>(gather);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionGatherDetectionNegative2) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto input2 = std::make_shared<Parameter>(element::i64, Shape{3});
        auto gather = std::make_shared<Gather>(input, input2, Constant::create(element::i64, {}, {1}));
        auto relu = std::make_shared<Relu>(gather);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input, input2 });
        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionGatherDetectionNegative3) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto gather = create_gather(input, {1}, 1);
        auto relu = std::make_shared<Relu>(gather);
        auto conv = create_conv(relu, {6, 1, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionGatherDetectionNegative4) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto gather = create_gather(input, {2, 0}, 1);
        auto relu = std::make_shared<Relu>(gather);
        auto conv = create_conv(relu, {6, 2, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionSplitConcatDetectionNegative) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto split = std::make_shared<Split>(input, Constant::create(element::i64, {}, {1}), 3);
        auto concat = std::make_shared<Concat>(OutputVector{split->output(2), split->output(1), split->output(1)}, 1);
        auto relu = std::make_shared<Relu>(concat);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionSplitConcatDetectionNegative2) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto split = std::make_shared<Split>(input, Constant::create(element::i64, {}, {1}), 3);
        auto split2 = std::make_shared<Split>(input, Constant::create(element::i64, {}, {1}), 3);
        auto concat = std::make_shared<Concat>(OutputVector{split->output(2), split->output(1), split2->output(0)}, 1);
        auto relu = std::make_shared<Relu>(concat);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, RICFusionSplitConcatDetectionNegative3) {
    {
        auto input = create_param({ 1, 3, 64, 64 });
        auto split = std::make_shared<Split>(input, Constant::create(element::i64, {}, {1}), 3);
        auto concat = std::make_shared<Concat>(OutputVector{split->output(2), split->output(1), split->output(0)}, 1);
        auto relu = std::make_shared<Relu>(concat);
        auto conv = create_conv(relu, {6, 3, 3, 3});

        function = std::make_shared<Function>(OutputVector{ conv, split->output(0) }, ParameterVector{ input });
        manager.register_pass<pass::ReverseInputChannelsFusion>();
    }
}

TEST_F(TransformationTestsF, FuseConvertLayout) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 64 });
        auto order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 3 }, { 1, 2, 0 });
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, order);
        auto relu = std::make_shared<ngraph::opset6::Relu>(transpose);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ relu }, ngraph::ParameterVector{ input });

        using namespace ov::preprocess;
        PrePostProcessor p(function);
        p.input(0).tensor().set_element_type(element::f16);
        p.input(0).preprocess().convert_layout({2, 0, 1});
        p.build();

        manager.register_pass<ngraph::pass::TransposeSinking>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f16, ngraph::Shape{ 3, 64, 1 });
        auto convert = std::make_shared<ngraph::opset6::Convert>(input, element::f32);
        auto relu = std::make_shared<ngraph::opset6::Relu>(convert);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ relu }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, FuseScaleValue) {
    {
        auto input = create_param({ 1, 64, 64, 3 });
        auto order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 });
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, order);
        auto conv = create_conv(transpose, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });

        using namespace ov::preprocess;
        PrePostProcessor p(function);
        p.input(0).tensor().set_layout("NHWC");
        p.input(0).preprocess().scale(1.3);
        p.build();

        manager.register_pass<pass::MOCTransformations>(false);
    }

    {
        auto input = create_param({ 1, 64, 64, 3 });
        auto order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 });
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, order);
        auto conv = create_conv(transpose, {6, 3, 3, 3});
        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, FuseScaleValues) {
    {
        auto input = create_param({ 1, 64, 64, 3 });
        auto order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 });
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, order);
        auto conv = create_conv(transpose, {6, 3, 3, 3});

        function = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });

        using namespace ov::preprocess;
        PrePostProcessor p(function);
        p.input(0).tensor().set_layout("NHWC");
        p.input(0).preprocess().scale({1.3, 0.2, 4.1});
        p.build();

        manager.register_pass<pass::MOCTransformations>(false);
    }

    {
        auto input = create_param({ 1, 64, 64, 3 });
        auto order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 });
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, order);
        auto conv = create_conv(transpose, {6, 3, 3, 3});
        function_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
