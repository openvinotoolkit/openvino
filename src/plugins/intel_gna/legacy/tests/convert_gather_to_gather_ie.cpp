// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <legacy/ngraph_ops/gather_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_gather_to_gather_ie.hpp>
#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertGatherToGatherIEStatic1) {
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{15, 4, 20, 28});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        model = std::make_shared<Model>(NodeVector{gather}, ParameterVector{input, indices});
        manager.register_pass<ngraph::pass::ConvertGatherToGatherIEMatcher>();
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{15, 4, 20, 28});
        auto gather = std::make_shared<ngraph::op::GatherIE>(input, indices, 1);

        model_ref = std::make_shared<Model>(NodeVector{gather}, ParameterVector{input, indices});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToGatherIEStatic2) {
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        model = std::make_shared<Model>(NodeVector{gather}, ParameterVector{input, indices});
        manager.register_pass<ngraph::pass::ConvertGatherToGatherIEMatcher>();
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto unsqueeze =
            std::make_shared<opset1::Unsqueeze>(indices, opset1::Constant::create(element::i64, Shape{1}, {0}));
        auto gather = std::make_shared<ngraph::op::GatherIE>(input, unsqueeze, 1);
        auto squeeze = std::make_shared<opset1::Squeeze>(gather, opset1::Constant::create(element::i64, Shape{1}, {1}));

        model_ref = std::make_shared<Model>(NodeVector{squeeze}, ParameterVector{input, indices});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToGatherIEDynamic1) {
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN, DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        model = std::make_shared<Model>(NodeVector{gather}, ParameterVector{input, indices});
        manager.register_pass<ngraph::pass::ConvertGatherToGatherIEMatcher>();
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN, DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto gather = std::make_shared<ngraph::op::GatherIE>(input, indices, 1);

        model_ref = std::make_shared<Model>(NodeVector{gather}, ParameterVector{input, indices});
    }
}

TEST_F(TransformationTestsF, ConvertGatherToGatherIEDynamic2) {
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        model = std::make_shared<Model>(NodeVector{gather}, ParameterVector{input, indices});
        manager.register_pass<ngraph::pass::ConvertGatherToGatherIEMatcher>();
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto unsqueeze =
            std::make_shared<opset1::Unsqueeze>(indices, opset1::Constant::create(element::i64, Shape{1}, {0}));
        auto gather = std::make_shared<ngraph::op::GatherIE>(input, unsqueeze, 1);
        auto squeeze = std::make_shared<opset1::Squeeze>(gather, opset1::Constant::create(element::i64, Shape{1}, {1}));

        model_ref = std::make_shared<Model>(NodeVector{squeeze}, ParameterVector{input, indices});
    }
}
