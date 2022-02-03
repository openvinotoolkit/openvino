// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_gather_to_gather_ie.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <legacy/ngraph_ops/gather_ie.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, ConvertGatherToGatherIEStatic1) {
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{15, 4, 20, 28});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        function = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});
        manager.register_pass<pass::ConvertGatherToGatherIEMatcher>();
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{15, 4, 20, 28});
        auto gather = std::make_shared<op::GatherIE>(input, indices, 1);

        function_ref = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToGatherIEStatic2) {
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        function = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});
        manager.register_pass<pass::ConvertGatherToGatherIEMatcher>();
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto unsqueeze = std::make_shared<opset1::Unsqueeze>(indices, opset1::Constant::create(element::i64, Shape{1}, {0}));
        auto gather = std::make_shared<op::GatherIE>(input, unsqueeze, 1);
        auto squeeze = std::make_shared<opset1::Squeeze>(gather, opset1::Constant::create(element::i64, Shape{1}, {1}));

        function_ref = std::make_shared<Function>(NodeVector{squeeze}, ParameterVector{input, indices});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToGatherIEDynamic1) {
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN, DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        function = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});
        manager.register_pass<pass::ConvertGatherToGatherIEMatcher>();
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN, DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto gather = std::make_shared<op::GatherIE>(input, indices, 1);

        function_ref = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToGatherIEDynamic2) {
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        function = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});
        manager.register_pass<pass::ConvertGatherToGatherIEMatcher>();
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto unsqueeze = std::make_shared<opset1::Unsqueeze>(indices, opset1::Constant::create(element::i64, Shape{1}, {0}));
        auto gather = std::make_shared<op::GatherIE>(input, unsqueeze, 1);
        auto squeeze = std::make_shared<opset1::Squeeze>(gather, opset1::Constant::create(element::i64, Shape{1}, {1}));

        function_ref = std::make_shared<Function>(NodeVector{squeeze}, ParameterVector{input, indices});
    }
}