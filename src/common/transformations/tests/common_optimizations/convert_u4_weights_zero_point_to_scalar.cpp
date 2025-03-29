// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_u4_weights_zero_point_to_scalar.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertU4WeightsFloatZeroPointToScalar) {
    auto weights_precision = ov::element::u4;
    auto decompression_precision = ov::element::f32;
    ov::Shape weights_shape{32, 128, 64};
    ov::Shape decompression_shape{32, 1, 64};
    {
        auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
        auto zero_point = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {8.1f});
        auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point);
        auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        model = std::make_shared<Model>(NodeVector{multiply}, ParameterVector{});
        manager.register_pass<ov::pass::ConvertU4WeightsZeroPointToScalar>();
    }
    {
        ov::Shape scalar_shape{};
        auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
        auto zero_point = ov::op::v0::Constant::create(decompression_precision, scalar_shape, {8.1f});
        auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point);
        auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        model_ref = std::make_shared<Model>(NodeVector{multiply}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::ACCURACY);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertU4WeightsU4ZeroPointToScalar) {
    auto weights_precision = ov::element::u4;
    auto decompression_precision = ov::element::f32;
    ov::Shape weights_shape{32, 128, 64};
    ov::Shape decompression_shape{32, 1, 64};
    {
        auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
        auto zero_point = ov::op::v0::Constant::create(weights_precision, decompression_shape, {8});
        auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, decompression_precision);
        auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point_convert);
        auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        model = std::make_shared<Model>(NodeVector{multiply}, ParameterVector{});
        manager.register_pass<ov::pass::ConvertU4WeightsZeroPointToScalar>();
    }
    {
        ov::Shape scalar_shape{};
        auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
        auto zero_point = ov::op::v0::Constant::create(weights_precision, scalar_shape, {8});
        auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, decompression_precision);
        auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point_convert);
        auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        model_ref = std::make_shared<Model>(NodeVector{multiply}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::ACCURACY);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertU4WeightsFloatZeroPointToScalarWeightsWithBiggerRank) {
    auto weights_precision = ov::element::u4;
    auto decompression_precision = ov::element::f32;
    ov::Shape weights_shape{32, 128, 64};
    ov::Shape decompression_shape{64};
    {
        auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
        auto zero_point = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {8});
        auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point);
        auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        model = std::make_shared<Model>(NodeVector{multiply}, ParameterVector{});
        manager.register_pass<ov::pass::ConvertU4WeightsZeroPointToScalar>();
    }
    {
        ov::Shape scalar_shape{};
        auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
        auto zero_point = ov::op::v0::Constant::create(decompression_precision, scalar_shape, {8});
        auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point);
        auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        model_ref = std::make_shared<Model>(NodeVector{multiply}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::ACCURACY);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, FuseU4WeightsAndZeroPointNotScalarLikeZP) {
    auto weights_precision = ov::element::u8;
    auto decompression_precision = ov::element::f32;
    ov::Shape weights_shape{32, 128, 64};
    ov::Shape decompression_shape{32, 1, 64};
    auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
    std::vector<int> zero_point_values(ov::shape_size(decompression_shape), 8);
    zero_point_values.back() = 6;
    auto zero_point = ov::op::v0::Constant::create(weights_precision, decompression_shape, zero_point_values);
    auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, decompression_precision);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point_convert);
    auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    model = std::make_shared<Model>(NodeVector{multiply}, ParameterVector{});
    manager.register_pass<ov::pass::ConvertU4WeightsZeroPointToScalar>();
}

TEST_F(TransformationTestsF, FuseU4WeightsAndZeroPointNotU4Weights) {
    auto weights_precision = ov::element::u8;
    auto decompression_precision = ov::element::f32;
    ov::Shape weights_shape{32, 128, 64};
    ov::Shape decompression_shape{32, 1, 64};
    auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
    auto zero_point = ov::op::v0::Constant::create(weights_precision, decompression_shape, {8});
    auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, decompression_precision);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point_convert);
    auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    model = std::make_shared<Model>(NodeVector{multiply}, ParameterVector{});
    manager.register_pass<ov::pass::ConvertU4WeightsZeroPointToScalar>();
}

TEST_F(TransformationTestsF, ConvertU4WeightsFloatZeroPointToScalarAdditionalZPConsumer) {
    auto weights_precision = ov::element::u4;
    auto decompression_precision = ov::element::f32;
    ov::Shape weights_shape{32, 128, 64};
    ov::Shape decompression_shape{32, 1, 64};
    auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
    auto zero_point = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {8});
    auto zero_point_consumer = std::make_shared<ov::op::v3::ShapeOf>(zero_point);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point);
    auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    model = std::make_shared<Model>(NodeVector{multiply, zero_point_consumer}, ParameterVector{});
    manager.register_pass<ov::pass::ConvertU4WeightsZeroPointToScalar>();
}

TEST_F(TransformationTestsF, ConvertU4WeightsU4ZeroPointToScalarAdditionalZPConsumer) {
    auto weights_precision = ov::element::u4;
    auto decompression_precision = ov::element::f32;
    ov::Shape weights_shape{32, 128, 64};
    ov::Shape decompression_shape{32, 1, 64};
    auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
    auto zero_point = ov::op::v0::Constant::create(weights_precision, decompression_shape, {8});
    auto zero_point_consumer = std::make_shared<ov::op::v3::ShapeOf>(zero_point);
    auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, decompression_precision);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point_convert);
    auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    model = std::make_shared<Model>(NodeVector{multiply, zero_point_consumer}, ParameterVector{});
    manager.register_pass<ov::pass::ConvertU4WeightsZeroPointToScalar>();
}

TEST_F(TransformationTestsF, ConvertU4WeightsU4ZeroPointToScalarAdditionalZPConvertConsumer) {
    auto weights_precision = ov::element::u4;
    auto decompression_precision = ov::element::f32;
    ov::Shape weights_shape{32, 128, 64};
    ov::Shape decompression_shape{32, 1, 64};
    auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
    auto zero_point = ov::op::v0::Constant::create(weights_precision, decompression_shape, {8});
    auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, decompression_precision);
    auto zero_point_convert_consumer = std::make_shared<ov::op::v3::ShapeOf>(zero_point_convert);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point_convert);
    auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    model = std::make_shared<Model>(NodeVector{multiply, zero_point_convert_consumer}, ParameterVector{});
    manager.register_pass<ov::pass::ConvertU4WeightsZeroPointToScalar>();
}

TEST_F(TransformationTestsF, ConvertU4WeightsU4ZeroPointToScalarZPWithBiggerRank) {
    auto weights_precision = ov::element::u4;
    auto decompression_precision = ov::element::f32;
    ov::Shape weights_shape{32, 128, 64};
    ov::Shape decompression_shape{1, 32, 1, 64};
    auto weights = ov::op::v0::Constant::create(weights_precision, weights_shape, {4});
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, decompression_precision);
    auto zero_point = ov::op::v0::Constant::create(weights_precision, decompression_shape, {8});
    auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, decompression_precision);
    auto zero_point_convert_consumer = std::make_shared<ov::op::v3::ShapeOf>(zero_point_convert);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point_convert);
    auto scale = ov::op::v0::Constant::create(decompression_precision, decompression_shape, {3.f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    model = std::make_shared<Model>(NodeVector{multiply, zero_point_convert_consumer}, ParameterVector{});
    manager.register_pass<ov::pass::ConvertU4WeightsZeroPointToScalar>();
}
