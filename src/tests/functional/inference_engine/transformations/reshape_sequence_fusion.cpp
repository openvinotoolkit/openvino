// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <transformations/common_optimizations/reshape_sequence_fusion.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;

namespace {
Output<Node> reshape(Output<Node> input, std::vector<int64_t> values, bool special_zero = true) {
    return std::make_shared<opset6::Reshape>(input,
                opset6::Constant::create(element::i64, Shape{values.size()}, values), special_zero);
}
}

TEST_F(TransformationTestsF, ReshapeSequenceFusion1) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {3, 2});
        auto b = reshape(a, {2, 3});
        auto c = reshape(b, {6});
        function = std::make_shared<Function>(OutputVector{c}, ParameterVector{data});

        manager.register_pass<pass::ReshapeSequenceFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto c = reshape(data, {6});
        function_ref = std::make_shared<Function>(OutputVector{c}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusion2) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {3, 2});
        auto b = reshape(a, {6});
        function = std::make_shared<Function>(OutputVector{b}, ParameterVector{data});

        manager.register_pass<pass::ReshapeSequenceFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto c = reshape(data, {6});
        function_ref = std::make_shared<Function>(OutputVector{c}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusionNeg1) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {-1, 2});
        auto b = reshape(a, {6});
        function = std::make_shared<Function>(OutputVector{b}, ParameterVector{data});

        manager.register_pass<pass::ReshapeSequenceFusion>();
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusionNeg2) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {-1, 3});
        auto b = reshape(a, {6});
        function = std::make_shared<Function>(OutputVector{b}, ParameterVector{data});

        manager.register_pass<pass::ReshapeSequenceFusion>();
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusionNeg3) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {2, 3});
        auto b = reshape(a, {6});
        function = std::make_shared<Function>(OutputVector{a, b}, ParameterVector{data});

        manager.register_pass<pass::ReshapeSequenceFusion>();
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusionNeg4) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {2, 3});
        auto b = reshape(a, {0, 3});
        function = std::make_shared<Function>(OutputVector{b}, ParameterVector{data});

        manager.register_pass<pass::ReshapeSequenceFusion>();
    }
}
