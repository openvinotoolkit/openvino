// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/opsets/opset8.hpp>
#include <transformations/common_optimizations/align_eltwise_input_ranks.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


using AlignEltwiseInputRanksParams = std::tuple<PartialShape, Shape, Shape, bool>;

class AlignEltwiseInputRanksTest
        : public testing::WithParamInterface<AlignEltwiseInputRanksParams>,
          public TransformationTestsF {
};

TEST_P(AlignEltwiseInputRanksTest, FusionTest) {
    auto params = GetParam();
    const auto& input_shape = std::get<0>(params);
    auto const_shape = std::get<1>(params);
    auto expected_const_shape = std::get<2>(params);
    bool can_align = std::get<3>(params);

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto add = std::make_shared<opset8::Add>(data, op::Constant::create(element::f32, const_shape, {3}));
        auto less = std::make_shared<opset8::Less>(data, op::Constant::create(element::f32, const_shape, {5}));
        auto sqr_diff = std::make_shared<opset8::SquaredDifference>(data, op::Constant::create(element::f32, const_shape, {5}));
        auto convert = std::make_shared<opset8::Convert>(data, element::boolean);
        auto logical_or = std::make_shared<opset8::LogicalOr>(convert, op::Constant::create(element::boolean, const_shape, {false}));
        auto low = op::Constant::create(element::f32, const_shape, {0});
        auto high = op::Constant::create(element::f32, const_shape, {20});
        auto fq = std::make_shared<opset8::FakeQuantize>(add, low, high, low, high, 256);
        function = std::make_shared<Function>(NodeVector{less, logical_or, fq}, ParameterVector{data});

        manager.register_pass<pass::AlignEltwiseInputRanks>();
    }

    if (can_align) {
        auto data = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto add = std::make_shared<opset8::Add>(data, op::Constant::create(element::f32, expected_const_shape, {3}));
        auto less = std::make_shared<opset8::Less>(data, op::Constant::create(element::f32, expected_const_shape, {5}));
        auto sqr_diff = std::make_shared<opset8::SquaredDifference>(data, op::Constant::create(element::f32, expected_const_shape, {5}));
        auto convert = std::make_shared<opset8::Convert>(data, element::boolean);
        auto logical_or = std::make_shared<opset8::LogicalOr>(convert, op::Constant::create(element::boolean, expected_const_shape, {false}));
        auto low = op::Constant::create(element::f32, expected_const_shape, {0});
        auto high = op::Constant::create(element::f32, expected_const_shape, {20});
        auto fq = std::make_shared<opset8::FakeQuantize>(add, low, high, low, high, 256);
        function_ref = std::make_shared<Function>(NodeVector{less, logical_or, fq}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

static std::vector<AlignEltwiseInputRanksParams> params = {
    AlignEltwiseInputRanksParams(PartialShape::dynamic(3), {}, {1, 1, 1}, true),
    AlignEltwiseInputRanksParams(PartialShape::dynamic(3), {1}, {1, 1, 1}, true),
    AlignEltwiseInputRanksParams(PartialShape::dynamic(3), {1, 1}, {1, 1, 1}, true),
    AlignEltwiseInputRanksParams(PartialShape::dynamic(3), {3}, {1, 1, 3}, true),
    AlignEltwiseInputRanksParams(PartialShape::dynamic(3), {2, 3}, {1, 2, 3}, true),
    AlignEltwiseInputRanksParams(PartialShape::dynamic(4), {}, {1, 1, 1, 1}, true),
    AlignEltwiseInputRanksParams(PartialShape::dynamic(4), {3, 1, 1}, {1, 3, 1, 1}, true),
    AlignEltwiseInputRanksParams(PartialShape::dynamic(4), {3}, {1, 1, 1, 3}, true),
    AlignEltwiseInputRanksParams(Shape{1, 4, 10, 10}, {4, 1, 1}, {1, 4, 1, 1}, true),
    // negative cases
    AlignEltwiseInputRanksParams(PartialShape::dynamic(), {2, 3, 4}, {}, false),
    AlignEltwiseInputRanksParams(Shape{}, {}, {}, false),
    AlignEltwiseInputRanksParams(Shape{}, {1}, {}, false),
    AlignEltwiseInputRanksParams(Shape{}, {2, 3, 4}, {}, false),
};

INSTANTIATE_TEST_SUITE_P(TransformationTests, AlignEltwiseInputRanksTest, ::testing::ValuesIn(params));
