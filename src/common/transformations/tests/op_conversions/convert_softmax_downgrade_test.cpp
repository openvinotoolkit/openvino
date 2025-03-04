// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_softmax_downgrade.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, ConvertSoftMax8ToSoftMax1) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        int64_t axis = 1;
        auto softmax_8 = std::make_shared<opset8::Softmax>(data, axis);

        model = std::make_shared<ov::Model>(NodeVector{softmax_8}, ParameterVector{data});
        manager.register_pass<ov::pass::ConvertSoftMax8ToSoftMax1>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        size_t axis = 1;
        auto softmax_1 = std::make_shared<opset1::Softmax>(data, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{softmax_1}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertSoftMax8ToSoftMax1_negative_axis) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        int64_t axis = -1;
        auto softmax_8 = std::make_shared<opset8::Softmax>(data, axis);

        model = std::make_shared<ov::Model>(NodeVector{softmax_8}, ParameterVector{data});
        manager.register_pass<ov::pass::ConvertSoftMax8ToSoftMax1>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        size_t axis = 1;
        auto softmax_1 = std::make_shared<opset1::Softmax>(data, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{softmax_1}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertSoftMax8ToSoftMax1_input_rank_5) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3, 5, 5, 5});
        int64_t axis = -2;
        auto softmax_8 = std::make_shared<opset8::Softmax>(data, axis);

        model = std::make_shared<ov::Model>(NodeVector{softmax_8}, ParameterVector{data});
        manager.register_pass<ov::pass::ConvertSoftMax8ToSoftMax1>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3, 5, 5, 5});
        size_t axis = 3;
        auto softmax_1 = std::make_shared<opset1::Softmax>(data, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{softmax_1}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, negative_ConvertSoftMax8ToSoftMax1_dynamic_rank) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        int64_t axis = -3;
        auto softmax_8 = std::make_shared<opset8::Softmax>(data, axis);

        model = std::make_shared<ov::Model>(NodeVector{softmax_8}, ParameterVector{data});
        manager.register_pass<ov::pass::ConvertSoftMax8ToSoftMax1>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        int64_t axis = -3;
        auto softmax_8 = std::make_shared<opset8::Softmax>(data, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{softmax_8}, ParameterVector{data});
    }
}
