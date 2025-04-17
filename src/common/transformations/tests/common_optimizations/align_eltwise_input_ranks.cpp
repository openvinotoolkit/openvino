// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/align_eltwise_input_ranks.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/opsets/opset8_decl.hpp"

using namespace testing;
using namespace ov;

using AlignEltwiseInputRanksParams = std::tuple<PartialShape, Shape, Shape, bool>;

class AlignEltwiseInputRanksTestP : public testing::WithParamInterface<AlignEltwiseInputRanksParams>,
                                    public TransformationTestsF {};

TEST_P(AlignEltwiseInputRanksTestP, FusionTest) {
    auto params = GetParam();
    const auto& input_shape = std::get<0>(params);
    auto const_shape = std::get<1>(params);
    auto expected_const_shape = std::get<2>(params);
    bool can_align = std::get<3>(params);

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto add = std::make_shared<opset8::Add>(data, op::v0::Constant::create(element::f32, const_shape, {3}));
        auto less = std::make_shared<opset8::Less>(data, op::v0::Constant::create(element::f32, const_shape, {5}));
        auto sqr_diff =
            std::make_shared<opset8::SquaredDifference>(data, op::v0::Constant::create(element::f32, const_shape, {5}));
        auto convert = std::make_shared<opset8::Convert>(data, element::boolean);
        auto logical_or =
            std::make_shared<opset8::LogicalOr>(convert,
                                                op::v0::Constant::create(element::boolean, const_shape, {false}));
        auto low = op::v0::Constant::create(element::f32, const_shape, {0});
        auto high = op::v0::Constant::create(element::f32, const_shape, {20});
        auto fq = std::make_shared<opset8::FakeQuantize>(add, low, high, low, high, 256);
        model = std::make_shared<Model>(OutputVector{less, logical_or, fq}, ParameterVector{data});

        manager.register_pass<ov::pass::AlignEltwiseInputRanks>();
    }

    if (can_align) {
        auto data = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto add =
            std::make_shared<opset8::Add>(data, op::v0::Constant::create(element::f32, expected_const_shape, {3}));
        auto less =
            std::make_shared<opset8::Less>(data, op::v0::Constant::create(element::f32, expected_const_shape, {5}));
        auto sqr_diff = std::make_shared<opset8::SquaredDifference>(
            data,
            op::v0::Constant::create(element::f32, expected_const_shape, {5}));
        auto convert = std::make_shared<opset8::Convert>(data, element::boolean);
        auto logical_or = std::make_shared<opset8::LogicalOr>(
            convert,
            op::v0::Constant::create(element::boolean, expected_const_shape, {false}));
        auto low = op::v0::Constant::create(element::f32, expected_const_shape, {0});
        auto high = op::v0::Constant::create(element::f32, expected_const_shape, {20});
        auto fq = std::make_shared<opset8::FakeQuantize>(add, low, high, low, high, 256);
        model_ref = std::make_shared<Model>(OutputVector{less, logical_or, fq}, ParameterVector{data});
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

INSTANTIATE_TEST_SUITE_P(TransformationTests, AlignEltwiseInputRanksTestP, ::testing::ValuesIn(params));

class AlignEltwiseInputRanksTestF : public TransformationTestsF {};

TEST_F(AlignEltwiseInputRanksTestF, NegativeFakeQuantizeWithScalarFirstInput) {
    {
        auto data = op::v0::Constant::create(element::f32, Shape{}, {10});
        auto low = op::v0::Constant::create(element::f32, Shape{1}, {0});
        auto high = op::v0::Constant::create(element::f32, Shape{1}, {20});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, low, high, low, high, 256);
        model = std::make_shared<Model>(fq->outputs());

        manager.register_pass<ov::pass::AlignEltwiseInputRanks>();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
