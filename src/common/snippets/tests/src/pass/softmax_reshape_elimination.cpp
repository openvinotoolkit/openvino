// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/softmax_reshape_elimination.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, SoftmaxV1ReshapeElimination) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 240});
        auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{6, 240});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(data, shape0, false);
        auto softmax_v1 = std::make_shared<ov::op::v1::Softmax>(reshape0, 1);
        auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{2, 3, 240});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(softmax_v1, shape1, false);
        model = std::make_shared<Model>(NodeVector{reshape1}, ParameterVector{data});

        manager.register_pass<snippets::pass::SoftmaxReshapeElimination>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 240});
        auto softmax_v1 = std::make_shared<ov::op::v1::Softmax>(data, 2);
        model_ref = std::make_shared<Model>(NodeVector{softmax_v1}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SoftmaxV8ReshapeElimination) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 340, 240});
        auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{680, 240});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(data, shape0, false);
        auto softmax_v1 = std::make_shared<ov::op::v8::Softmax>(reshape0, -1);
        auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{1, 2, 340, 240});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(softmax_v1, shape1, false);
        model = std::make_shared<Model>(NodeVector{reshape1}, ParameterVector{data});

        manager.register_pass<snippets::pass::SoftmaxReshapeElimination>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 340, 240});
        auto softmax_v1 = std::make_shared<ov::op::v8::Softmax>(data, 3);
        model_ref = std::make_shared<Model>(NodeVector{softmax_v1}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SoftmaxReshapeElimination_IncorrectReshape) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 340, 240});
        auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{2, 81600});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(data, shape0, false);
        auto softmax_v1 = std::make_shared<ov::op::v8::Softmax>(reshape0, -1);
        auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{1, 2, 340, 240});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(softmax_v1, shape1, false);
        model = std::make_shared<Model>(NodeVector{reshape1}, ParameterVector{data});

        manager.register_pass<snippets::pass::SoftmaxReshapeElimination>();
    }
}

TEST_F(TransformationTestsF, SoftmaxV8ReshapeElimination_DynamicBatch) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::PartialShape{-1, 2, 340, 240});
        auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{-1, 240});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(data, shape0, false);
        auto softmax_v1 = std::make_shared<ov::op::v8::Softmax>(reshape0, -1);
        auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{-1, 2, 340, 240});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(softmax_v1, shape1, false);
        model = std::make_shared<Model>(NodeVector{reshape1}, ParameterVector{data});

        manager.register_pass<snippets::pass::SoftmaxReshapeElimination>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::PartialShape{-1, 2, 340, 240});
        auto softmax_v1 = std::make_shared<ov::op::v8::Softmax>(data, 3);
        model_ref = std::make_shared<Model>(NodeVector{softmax_v1}, ParameterVector{data});
    }
}
