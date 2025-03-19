// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "low_precision/add.hpp"
#include "low_precision/avg_pool.hpp"
#include "low_precision/clamp.hpp"
#include "low_precision/concat.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/depth_to_space.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/interpolate.hpp"
#include "low_precision/low_precision.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/max_pool.hpp"
#include "low_precision/multiply_partial.hpp"
#include "low_precision/mvn.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/normalize_l2.hpp"
#include "low_precision/pad.hpp"
#include "low_precision/prelu.hpp"
#include "low_precision/reduce_max.hpp"
#include "low_precision/reduce_mean.hpp"
#include "low_precision/reduce_min.hpp"
#include "low_precision/reduce_sum.hpp"
#include "low_precision/relu.hpp"
#include "low_precision/reshape.hpp"
#include "low_precision/shuffle_channels.hpp"
#include "low_precision/split.hpp"
#include "low_precision/squeeze.hpp"
#include "low_precision/strided_slice.hpp"
#include "low_precision/transpose.hpp"
#include "low_precision/unsqueeze.hpp"
#include "low_precision/variadic_split.hpp"
#include "ov_lpt_models/common/builders.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

TEST(LPT, AvoidDequantizationToShapeOfPropagationAddTransformation) {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto input2 = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});

    auto convert1 = std::make_shared<ov::op::v0::Convert>(input1, element::f32);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(input2, element::f32);

    auto mul1 = std::make_shared<ov::op::v1::Multiply>(convert1, ov::op::v0::Constant::create(element::f32, {}, {2.f}));
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(convert2, ov::op::v0::Constant::create(element::f32, {}, {4.f}));

    auto add = std::make_shared<ov::op::v1::Add>(mul1, mul2);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(add);

    auto result1 = std::make_shared<ov::op::v0::Result>(add);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input1, input2});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::AddTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationAvgPoolTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto avgPool =
        std::make_shared<ov::op::v1::AvgPool>(mul, Strides{1, 1}, Shape{1, 1}, Shape{0, 0}, Shape{2, 2}, true);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(avgPool);

    auto result1 = std::make_shared<ov::op::v0::Result>(avgPool);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::TypeRelaxedReplacer>();
    m.register_pass<ov::pass::low_precision::AvgPoolTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationClampTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto clamp = std::make_shared<ov::op::v0::Clamp>(mul, 0.0, 6.0);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(clamp);

    auto result1 = std::make_shared<ov::op::v0::Result>(clamp);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::TypeRelaxedReplacer>();
    m.register_pass<ov::pass::low_precision::ClampTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationConcatTransformation) {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto input2 = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});

    auto convert1 = std::make_shared<ov::op::v0::Convert>(input1, element::f32);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(input2, element::f32);

    auto mul1 = std::make_shared<ov::op::v1::Multiply>(convert1, ov::op::v0::Constant::create(element::f32, {}, {2.f}));
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(convert2, ov::op::v0::Constant::create(element::f32, {}, {4.f}));

    auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{mul1, mul2}, 1);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(concat);

    auto result1 = std::make_shared<ov::op::v0::Result>(concat);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input1, input2});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::ConcatTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationConvolutionTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto weights = ov::op::v0::Constant::create(element::i8, {6, 3, 1, 1}, {3});
    auto convertOnWeights = std::make_shared<ov::op::v0::Convert>(weights, element::f32);
    auto mulOnWeights =
        std::make_shared<ov::op::v1::Multiply>(convertOnWeights, ov::op::v0::Constant::create(element::f32, {}, {4.f}));

    auto convolution = std::make_shared<ov::op::v1::Convolution>(mul,
                                                                 mulOnWeights,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});

    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(convolution);

    auto result1 = std::make_shared<ov::op::v0::Result>(convolution);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::ConvolutionTransformation>();
    m.run_passes(f);
    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationConvolutionBackpropDataTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 8, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto weights = ov::op::v0::Constant::create(element::i8, {8, 2, 1, 1}, {3});
    auto convertOnWeights = std::make_shared<ov::op::v0::Convert>(weights, element::f32);
    auto mulOnWeights =
        std::make_shared<ov::op::v1::Multiply>(convertOnWeights, ov::op::v0::Constant::create(element::f32, {}, {4.f}));

    auto convolutionBackpropData = std::make_shared<ov::op::v1::ConvolutionBackpropData>(mul,
                                                                                         mulOnWeights,
                                                                                         ov::Strides{1, 1},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::Strides{1, 1});

    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(convolutionBackpropData);

    auto result1 = std::make_shared<ov::op::v0::Result>(convolutionBackpropData);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::ConvolutionBackpropDataTransformation>();
    m.run_passes(f);
    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationDepthToSpaceTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto d2s = std::make_shared<ov::op::v0::DepthToSpace>(mul, op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(d2s);

    auto result1 = std::make_shared<ov::op::v0::Result>(d2s);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::DepthToSpaceTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationFakeQuantizeDecompositionTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 3, 16, 16});

    ov::builder::subgraph::FakeQuantizeOnData fqValues{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}};
    auto fakeQuantize = ov::builder::subgraph::makeFakeQuantize(input, element::f32, fqValues);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(fakeQuantize);

    auto& outInfo = fakeQuantize->output(0).get_rt_info();
    outInfo.emplace(ov::PrecisionsAttribute::get_type_info_static(),
                    ov::PrecisionsAttribute(element::TypeVector{element::u8, element::i8}));

    auto result1 = std::make_shared<ov::op::v0::Result>(fakeQuantize);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::FakeQuantizeDecompositionTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationGroupConvolutionTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 2 * 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto weights = ov::op::v0::Constant::create(element::i8, {6, 3, 7, 7}, {2});
    auto convertOnWeights = std::make_shared<ov::op::v0::Convert>(weights, element::f32);
    auto mulOnWeights =
        std::make_shared<ov::op::v1::Multiply>(convertOnWeights, ov::op::v0::Constant::create(element::f32, {}, {4.f}));
    auto reshapeConst = ov::op::v0::Constant::create(element::i32, {5}, {2, 3, 3, 7, 7});
    auto reshapeOnWeights = std::make_shared<ov::op::v1::Reshape>(mulOnWeights, reshapeConst, true);

    auto groupConvolution = std::make_shared<ov::op::v1::GroupConvolution>(mul,
                                                                           reshapeOnWeights,
                                                                           ov::Strides{1, 1},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::Strides{1, 1});
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(groupConvolution);

    auto result1 = std::make_shared<ov::op::v0::Result>(groupConvolution);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::GroupConvolutionTransformation>();
    m.run_passes(f);
    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationInterpolateTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto outShape = ov::op::v0::Constant::create(element::i32, {4}, {1, 3, 18, 18});
    op::v0::Interpolate::Attributes attributes;
    attributes.align_corners = false;
    attributes.antialias = false;
    attributes.axes = AxisSet{2, 3};
    attributes.mode = "nearest";
    attributes.pads_begin = {0};
    attributes.pads_end = {0};
    auto interpolate = std::make_shared<ov::op::v0::Interpolate>(mul, outShape, attributes);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(interpolate);

    auto result1 = std::make_shared<ov::op::v0::Result>(interpolate);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::InterpolateTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationMatMulTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 1024});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto weights = ov::op::v0::Constant::create(element::i8, {2048, 1024}, {3});
    auto convertOnWeights = std::make_shared<ov::op::v0::Convert>(weights, element::f32);
    auto mulOnWeights =
        std::make_shared<ov::op::v1::Multiply>(convertOnWeights, ov::op::v0::Constant::create(element::f32, {}, {4.f}));

    auto matmul = std::make_shared<ov::op::v0::MatMul>(mul, mulOnWeights, false, true);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(matmul);

    auto result1 = std::make_shared<ov::op::v0::Result>(matmul);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::MatMulTransformation>();
    m.run_passes(f);
    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationMaxPoolTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto maxPool = std::make_shared<ov::op::v1::MaxPool>(mul, Strides{1, 1}, Shape{1, 1}, Shape{0, 0}, Shape{2, 2});
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(maxPool);

    auto result1 = std::make_shared<ov::op::v0::Result>(maxPool);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::MaxPoolTransformation>();
    m.run_passes(f);
    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationMultiplyTransformation) {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto input2 = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});

    auto convert1 = std::make_shared<ov::op::v0::Convert>(input1, element::f32);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(input2, element::f32);

    auto mul1 = std::make_shared<ov::op::v1::Multiply>(convert1, ov::op::v0::Constant::create(element::f32, {}, {2.f}));
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(convert2, ov::op::v0::Constant::create(element::f32, {}, {4.f}));

    auto mul = std::make_shared<ov::op::v1::Multiply>(mul1, mul2);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(mul);

    auto result1 = std::make_shared<ov::op::v0::Result>(mul);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input1, input2});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::MultiplyPartialTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationMVNTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto MVN = std::make_shared<ov::op::TypeRelaxed<op::v0::MVN>>(mul);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(MVN);

    auto result1 = std::make_shared<ov::op::v0::Result>(MVN);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::MVNTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationNormalizeL2Transformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto axes = ov::op::v0::Constant::create(element::i32, {2}, {2, 3});
    auto normalize = std::make_shared<ov::op::v0::NormalizeL2>(mul, axes, 0.01f, ov::op::EpsMode::ADD);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(normalize);

    auto result1 = std::make_shared<ov::op::v0::Result>(normalize);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::NormalizeL2Transformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationPadTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto pads_begin = ov::op::v0::Constant::create(element::i32, {4}, {0, 0, 1, 1});
    auto pads_end = ov::op::v0::Constant::create(element::i32, {4}, {0, 0, 1, 1});
    auto pad = std::make_shared<ov::op::v1::Pad>(mul, pads_begin, pads_end, op::PadMode::CONSTANT);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(pad);

    auto result1 = std::make_shared<ov::op::v0::Result>(pad);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::PadTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationPReluTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto slope = ov::op::v0::Constant::create(element::f32, {1, 3, 1, 1}, {0.01f});
    auto prelu = std::make_shared<ov::op::v0::PRelu>(mul, slope);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(prelu);

    auto result1 = std::make_shared<ov::op::v0::Result>(prelu);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::TypeRelaxedReplacer>();
    m.register_pass<ov::pass::low_precision::PReluTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReduceMaxTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto axes = ov::op::v0::Constant::create(element::i32, {2}, {2, 3});
    auto reduce = std::make_shared<ov::op::v1::ReduceMax>(mul, axes);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(reduce);

    auto result1 = std::make_shared<ov::op::v0::Result>(reduce);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::ReduceMaxTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReduceMeanTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto axes = ov::op::v0::Constant::create(element::i32, {2}, {2, 3});
    auto reduce = std::make_shared<ov::op::v1::ReduceMean>(mul, axes);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(reduce);

    auto result1 = std::make_shared<ov::op::v0::Result>(reduce);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::TypeRelaxedReplacer>();
    m.register_pass<ov::pass::low_precision::ReduceMeanTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReduceMinTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto axes = ov::op::v0::Constant::create(element::i32, {2}, {2, 3});
    auto reduce = std::make_shared<ov::op::v1::ReduceMin>(mul, axes);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(reduce);

    auto result1 = std::make_shared<ov::op::v0::Result>(reduce);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::ReduceMinTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReduceSumTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto axes = ov::op::v0::Constant::create(element::i32, {2}, {2, 3});
    auto reduce = std::make_shared<ov::op::v1::ReduceSum>(mul, axes);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(reduce);

    auto result1 = std::make_shared<ov::op::v0::Result>(reduce);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::TypeRelaxedReplacer>();
    m.register_pass<ov::pass::low_precision::ReduceSumTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReshapeTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto outShape = ov::op::v0::Constant::create(element::i32, {3}, {1, 3, -1});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(mul, outShape, true);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(reshape);

    auto result1 = std::make_shared<ov::op::v0::Result>(reshape);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::ReshapeTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReluTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto relu = std::make_shared<ov::op::v0::Relu>(mul);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(relu);

    auto result1 = std::make_shared<ov::op::v0::Result>(relu);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::ReluTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationSqueezeTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto axes = ov::op::v0::Constant::create(element::i32, {1}, {0});
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(mul, axes);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(squeeze);

    auto result1 = std::make_shared<ov::op::v0::Result>(squeeze);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::SqueezeTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationSplitTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto axis = ov::op::v0::Constant::create(element::i32, {}, {1});
    auto split = std::make_shared<ov::op::v1::Split>(mul, axis, 3);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(split);

    auto result1 = std::make_shared<ov::op::v0::Result>(split->output(0));
    auto result2 = std::make_shared<ov::op::v0::Result>(split->output(1));
    auto result3 = std::make_shared<ov::op::v0::Result>(split->output(2));
    auto result4 = std::make_shared<ov::op::v0::Result>(shapeOf->output(0));

    auto f = std::make_shared<Model>(ResultVector{result1, result2, result3, result4}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::SplitTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationShuffleChannelsTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto shuffleChannels = std::make_shared<ov::op::v0::ShuffleChannels>(mul);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(shuffleChannels);

    auto result1 = std::make_shared<ov::op::v0::Result>(shuffleChannels);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::ShuffleChannelsTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationStridedSliceTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto beginParam = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 0, 0});
    auto endParam = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 2, 1, 1});
    auto stridesParam = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 1, 1, 1});
    auto stridedSlice = std::make_shared<ov::op::v1::StridedSlice>(mul,
                                                                   beginParam,
                                                                   endParam,
                                                                   stridesParam,
                                                                   std::vector<std::int64_t>{1, 0, 1, 1},
                                                                   std::vector<std::int64_t>{1, 0, 1, 1});
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(stridedSlice);

    auto result1 = std::make_shared<ov::op::v0::Result>(stridedSlice);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::StridedSliceTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationTransposeTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto constant = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(mul, constant);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(transpose);

    auto result1 = std::make_shared<ov::op::v0::Result>(transpose);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::TransposeTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationUnsqueezeTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto axes = ov::op::v0::Constant::create(element::i32, {1}, {3});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(mul, axes);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(unsqueeze);

    auto result1 = std::make_shared<ov::op::v0::Result>(unsqueeze);
    auto result2 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::UnsqueezeTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationVariadicSplitTransformation) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{1, 3, 16, 16});
    auto convert = std::make_shared<ov::op::v0::Convert>(input, element::f32);
    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, {}, {2.f}));

    auto axis = ov::op::v0::Constant::create(element::i32, {}, {1});
    auto lengths = ov::op::v0::Constant::create(element::i32, {2}, {1, 2});
    auto variadicSplit = std::make_shared<ov::op::v1::VariadicSplit>(mul, axis, lengths);
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(variadicSplit->output(0));

    auto result1 = std::make_shared<ov::op::v0::Result>(variadicSplit->output(0));
    auto result2 = std::make_shared<ov::op::v0::Result>(variadicSplit->output(1));
    auto result3 = std::make_shared<ov::op::v0::Result>(shapeOf);

    auto f = std::make_shared<Model>(ResultVector{result1, result2, result3}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::low_precision::VariadicSplitTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = ov::pass::low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}
