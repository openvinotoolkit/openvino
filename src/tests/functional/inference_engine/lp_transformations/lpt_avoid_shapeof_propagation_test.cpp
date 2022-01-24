// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <sstream>
#include <memory>

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
#include "low_precision/mat_mul.hpp"
#include "low_precision/max_pool.hpp"
#include "low_precision/multiply.hpp"
#include "low_precision/mvn.hpp"
#include "low_precision/normalize_l2.hpp"
#include "low_precision/pad.hpp"
#include "low_precision/prelu.hpp"
#include "low_precision/reduce_max.hpp"
#include "low_precision/reduce_mean.hpp"
#include "low_precision/reduce_min.hpp"
#include "low_precision/reduce_sum.hpp"
#include "low_precision/reshape.hpp"
#include "low_precision/relu.hpp"
#include "low_precision/squeeze.hpp"
#include "low_precision/split.hpp"
#include "low_precision/shuffle_channels.hpp"
#include "low_precision/strided_slice.hpp"
#include "low_precision/transpose.hpp"
#include "low_precision/unsqueeze.hpp"
#include "low_precision/variadic_split.hpp"

#include "low_precision/network_helper.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

#include <gtest/gtest.h>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

TEST(LPT, AvoidDequantizationToShapeOfPropagationAddTransformation) {
    auto input1 = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto input2 = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });

    auto convert1 = std::make_shared<opset1::Convert>(input1, element::f32);
    auto convert2 = std::make_shared<opset1::Convert>(input2, element::f32);

    auto mul1 = std::make_shared<opset1::Multiply>(convert1, opset1::Constant::create(element::f32, {}, { 2.f }));
    auto mul2 = std::make_shared<opset1::Multiply>(convert2, opset1::Constant::create(element::f32, {}, { 4.f }));

    auto add = std::make_shared<opset1::Add>(mul1, mul2);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(add);

    auto result1 = std::make_shared<opset1::Result>(add);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input1, input2 });
    pass::Manager m;
    m.register_pass<pass::low_precision::AddTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationAvgPoolTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto avgPool = std::make_shared<opset1::AvgPool>(mul, Strides{ 1, 1 }, Shape{ 1, 1 }, Shape{ 0, 0 }, Shape{ 2, 2 }, true);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(avgPool);

    auto result1 = std::make_shared<opset1::Result>(avgPool);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::AvgPoolTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationClampTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto clamp = std::make_shared<opset1::Clamp>(mul, 0.0, 6.0);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(clamp);

    auto result1 = std::make_shared<opset1::Result>(clamp);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::ClampTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationConcatTransformation) {
    auto input1 = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto input2 = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });

    auto convert1 = std::make_shared<opset1::Convert>(input1, element::f32);
    auto convert2 = std::make_shared<opset1::Convert>(input2, element::f32);

    auto mul1 = std::make_shared<opset1::Multiply>(convert1, opset1::Constant::create(element::f32, {}, { 2.f }));
    auto mul2 = std::make_shared<opset1::Multiply>(convert2, opset1::Constant::create(element::f32, {}, { 4.f }));

    auto concat = std::make_shared<opset1::Concat>(OutputVector{ mul1, mul2 }, 1);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(concat);

    auto result1 = std::make_shared<opset1::Result>(concat);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input1, input2 });
    pass::Manager m;
    m.register_pass<pass::low_precision::ConcatTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationConvolutionTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto weights = opset1::Constant::create(element::i8, { 6, 3, 1, 1 }, { 3 });
    auto convertOnWeights = std::make_shared<opset1::Convert>(weights, element::f32);
    auto mulOnWeights = std::make_shared<opset1::Multiply>(convertOnWeights, opset1::Constant::create(element::f32, {}, { 4.f }));

    auto convolution = std::make_shared<opset1::Convolution>(
        mul,
        mulOnWeights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    auto shapeOf = std::make_shared<opset1::ShapeOf>(convolution);

    auto result1 = std::make_shared<opset1::Result>(convolution);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::ConvolutionTransformation>();
    m.run_passes(f);
    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationConvolutionBackpropDataTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 8, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto weights = opset1::Constant::create(element::i8, { 8, 2, 1, 1 }, { 3 });
    auto convertOnWeights = std::make_shared<opset1::Convert>(weights, element::f32);
    auto mulOnWeights = std::make_shared<opset1::Multiply>(convertOnWeights, opset1::Constant::create(element::f32, {}, { 4.f }));

    auto convolutionBackpropData = std::make_shared<opset1::ConvolutionBackpropData>(
        mul,
        mulOnWeights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    auto shapeOf = std::make_shared<opset1::ShapeOf>(convolutionBackpropData);

    auto result1 = std::make_shared<opset1::Result>(convolutionBackpropData);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::ConvolutionBackpropDataTransformation>();
    m.run_passes(f);
    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationDepthToSpaceTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto d2s = std::make_shared<opset1::DepthToSpace>(mul, op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(d2s);

    auto result1 = std::make_shared<opset1::Result>(d2s);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::DepthToSpaceTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationFakeQuantizeDecompositionTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 16, 16 });

    ngraph::builder::subgraph::FakeQuantizeOnData fqValues{ 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} };
    auto fakeQuantize = ngraph::builder::subgraph::makeFakeQuantize(input, element::f32, fqValues);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(fakeQuantize);

    auto& outInfo = fakeQuantize->output(0).get_rt_info();
    outInfo.emplace(PrecisionsAttribute::get_type_info_static(), PrecisionsAttribute(element::TypeVector{ element::u8, element::i8 }));

    auto result1 = std::make_shared<opset1::Result>(fakeQuantize);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::FakeQuantizeDecompositionTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationGroupConvolutionTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 2 * 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto weights = opset1::Constant::create(element::i8, { 6, 3, 7, 7 }, { 2 });
    auto convertOnWeights = std::make_shared<opset1::Convert>(weights, element::f32);
    auto mulOnWeights = std::make_shared<opset1::Multiply>(convertOnWeights, opset1::Constant::create(element::f32, {}, { 4.f }));
    auto reshapeConst = opset1::Constant::create(element::i32, { 5 }, { 2, 3, 3, 7, 7 });
    auto reshapeOnWeights = std::make_shared<opset1::Reshape>(mulOnWeights, reshapeConst, true);

    auto groupConvolution = std::make_shared<opset1::GroupConvolution>(
        mul,
        reshapeOnWeights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    auto shapeOf = std::make_shared<opset1::ShapeOf>(groupConvolution);

    auto result1 = std::make_shared<opset1::Result>(groupConvolution);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::GroupConvolutionTransformation>();
    m.run_passes(f);
    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationInterpolateTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto outShape = opset1::Constant::create(element::i32, { 4 }, { 1, 3, 18, 18});
    op::v0::InterpolateAttrs attributes;
    attributes.align_corners = false;
    attributes.antialias = false;
    attributes.axes = AxisSet{ 2, 3 };
    attributes.mode = "nearest";
    attributes.pads_begin = { 0 };
    attributes.pads_end = { 0 };
    auto interpolate = std::make_shared<opset1::Interpolate>(mul, outShape, attributes);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(interpolate);

    auto result1 = std::make_shared<opset1::Result>(interpolate);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::InterpolateTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationMatMulTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 1024 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto weights = opset1::Constant::create(element::i8, { 2048, 1024 }, { 3 });
    auto convertOnWeights = std::make_shared<opset1::Convert>(weights, element::f32);
    auto mulOnWeights = std::make_shared<opset1::Multiply>(convertOnWeights, opset1::Constant::create(element::f32, {}, { 4.f }));

    auto matmul = std::make_shared<opset1::MatMul>(mul, mulOnWeights, false, true);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(matmul);

    auto result1 = std::make_shared<opset1::Result>(matmul);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::MatMulTransformation>();
    m.run_passes(f);
    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationMaxPoolTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto maxPool = std::make_shared<opset1::MaxPool>(mul, Strides{ 1, 1 }, Shape{ 1, 1 }, Shape{ 0, 0 }, Shape{ 2, 2 });
    auto shapeOf = std::make_shared<opset1::ShapeOf>(maxPool);

    auto result1 = std::make_shared<opset1::Result>(maxPool);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::MaxPoolTransformation>();
    m.run_passes(f);
    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationMultiplyTransformation) {
    auto input1 = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto input2 = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });

    auto convert1 = std::make_shared<opset1::Convert>(input1, element::f32);
    auto convert2 = std::make_shared<opset1::Convert>(input2, element::f32);

    auto mul1 = std::make_shared<opset1::Multiply>(convert1, opset1::Constant::create(element::f32, {}, { 2.f }));
    auto mul2 = std::make_shared<opset1::Multiply>(convert2, opset1::Constant::create(element::f32, {}, { 4.f }));

    auto mul = std::make_shared<opset1::Multiply>(mul1, mul2);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(mul);

    auto result1 = std::make_shared<opset1::Result>(mul);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input1, input2 });
    pass::Manager m;
    m.register_pass<pass::low_precision::MultiplyTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationMVNTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto MVN = std::make_shared<op::TypeRelaxed<op::v0::MVN>>(mul);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(MVN);

    auto result1 = std::make_shared<opset1::Result>(MVN);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::MVNTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationNormalizeL2Transformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto axes = opset1::Constant::create(element::i32, { 2 }, { 2, 3 });
    auto normalize = std::make_shared<opset1::NormalizeL2>(mul, axes, 0.01, ov::op::EpsMode::ADD);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(normalize);

    auto result1 = std::make_shared<opset1::Result>(normalize);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::NormalizeL2Transformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationPadTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto pads_begin = opset1::Constant::create(element::i32, { 4 }, { 0, 0, 1, 1 });
    auto pads_end = opset1::Constant::create(element::i32, { 4 }, { 0, 0, 1, 1 });
    auto pad = std::make_shared<opset1::Pad>(mul, pads_begin, pads_end, op::PadMode::CONSTANT);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(pad);

    auto result1 = std::make_shared<opset1::Result>(pad);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::PadTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationPReluTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto slope = opset1::Constant::create(element::f32, { 1, 3, 1, 1 }, { 0.01f });
    auto prelu = std::make_shared<opset1::PRelu>(mul, slope);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(prelu);

    auto result1 = std::make_shared<opset1::Result>(prelu);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::PReluTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReduceMaxTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto axes = opset1::Constant::create(element::i32, { 2 }, { 2, 3 });
    auto reduce = std::make_shared<opset1::ReduceMax>(mul, axes);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(reduce);

    auto result1 = std::make_shared<opset1::Result>(reduce);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::ReduceMaxTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReduceMeanTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto axes = opset1::Constant::create(element::i32, { 2 }, { 2, 3 });
    auto reduce = std::make_shared<opset1::ReduceMean>(mul, axes);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(reduce);

    auto result1 = std::make_shared<opset1::Result>(reduce);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::ReduceMeanTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReduceMinTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto axes = opset1::Constant::create(element::i32, { 2 }, { 2, 3 });
    auto reduce = std::make_shared<opset1::ReduceMin>(mul, axes);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(reduce);

    auto result1 = std::make_shared<opset1::Result>(reduce);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::ReduceMinTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReduceSumTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto axes = opset1::Constant::create(element::i32, { 2 }, { 2, 3 });
    auto reduce = std::make_shared<opset1::ReduceSum>(mul, axes);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(reduce);

    auto result1 = std::make_shared<opset1::Result>(reduce);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::ReduceSumTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReshapeTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto outShape = opset1::Constant::create(element::i32, { 3 }, { 1, 3, -1 });
    auto reshape = std::make_shared<opset1::Reshape>(mul, outShape, true);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(reshape);

    auto result1 = std::make_shared<opset1::Result>(reshape);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::ReshapeTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationReluTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto relu = std::make_shared<opset1::Relu>(mul);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(relu);

    auto result1 = std::make_shared<opset1::Result>(relu);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::ReluTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationSqueezeTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto axes = opset1::Constant::create(element::i32, { 1 }, { 0 });
    auto squeeze = std::make_shared<opset1::Squeeze>(mul, axes);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(squeeze);

    auto result1 = std::make_shared<opset1::Result>(squeeze);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::SqueezeTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationSplitTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto axis = opset1::Constant::create(element::i32, {}, { 1 });
    auto split = std::make_shared<opset1::Split>(mul, axis, 3);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(split);

    auto result1 = std::make_shared<opset1::Result>(split->output(0));
    auto result2 = std::make_shared<opset1::Result>(split->output(1));
    auto result3 = std::make_shared<opset1::Result>(split->output(2));
    auto result4 = std::make_shared<opset1::Result>(shapeOf->output(0));

    auto f = std::make_shared<Function>(ResultVector{ result1, result2, result3, result4 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::SplitTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationShuffleChannelsTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto shuffleChannels = std::make_shared<opset1::ShuffleChannels>(mul);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(shuffleChannels);

    auto result1 = std::make_shared<opset1::Result>(shuffleChannels);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::ShuffleChannelsTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationStridedSliceTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto beginParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 0, 0, 0 });
    auto endParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 1, 2, 1, 1 });
    auto stridesParam = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 1, 1, 1, 1 });
    auto stridedSlice = std::make_shared<ngraph::opset1::StridedSlice>(
        mul, beginParam, endParam, stridesParam,
        std::vector<std::int64_t>{ 1, 0, 1, 1 },
        std::vector<std::int64_t>{ 1, 0, 1, 1 });
    auto shapeOf = std::make_shared<opset1::ShapeOf>(stridedSlice);

    auto result1 = std::make_shared<opset1::Result>(stridedSlice);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::StridedSliceTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationTransposeTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto constant = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 1, 3, 2 });
    auto transpose = std::make_shared<ngraph::opset1::Transpose>(mul, constant);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(transpose);

    auto result1 = std::make_shared<opset1::Result>(transpose);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::TransposeTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationUnsqueezeTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto axes = opset1::Constant::create(element::i32, { 1 }, { 3 });
    auto unsqueeze = std::make_shared<opset1::Unsqueeze>(mul, axes);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(unsqueeze);

    auto result1 = std::make_shared<opset1::Result>(unsqueeze);
    auto result2 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::UnsqueezeTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}

TEST(LPT, AvoidDequantizationToShapeOfPropagationVariadicSplitTransformation) {
    auto input = std::make_shared<opset1::Parameter>(element::u8, PartialShape{ 1, 3, 16, 16 });
    auto convert = std::make_shared<opset1::Convert>(input, element::f32);
    auto mul = std::make_shared<opset1::Multiply>(convert, opset1::Constant::create(element::f32, {}, { 2.f }));

    auto axis = opset1::Constant::create(element::i32, {}, { 1 });
    auto lengths = opset1::Constant::create(element::i32, { 2 }, { 1, 2 });
    auto variadicSplit = std::make_shared<opset1::VariadicSplit>(mul, axis, lengths);
    auto shapeOf = std::make_shared<opset1::ShapeOf>(variadicSplit->output(0));

    auto result1 = std::make_shared<opset1::Result>(variadicSplit->output(0));
    auto result2 = std::make_shared<opset1::Result>(variadicSplit->output(1));
    auto result3 = std::make_shared<opset1::Result>(shapeOf);

    auto f = std::make_shared<Function>(ResultVector{ result1, result2, result3 }, ParameterVector{ input });
    pass::Manager m;
    m.register_pass<pass::low_precision::VariadicSplitTransformation>();
    m.run_passes(f);

    auto dqBeforeShapeOf = low_precision::NetworkHelper::getDequantization(result2->get_input_node_shared_ptr(0));
    ASSERT_TRUE(dqBeforeShapeOf.empty());
}
