// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convertlike.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace ov;

class ConvertConvertLikeTransformTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::ConvertConvertLike>();
    }
};

TEST_F(ConvertConvertLikeTransformTests, ConvertLikeWithConstant) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto like = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto cvtlike = std::make_shared<opset8::ConvertLike>(data, like);
        model = std::make_shared<ov::Model>(OutputVector{cvtlike}, ParameterVector{data});
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto cvt = std::make_shared<opset8::Convert>(data, element::i32);
        model_ref = std::make_shared<ov::Model>(OutputVector{cvt}, ParameterVector{data});
    }
}

TEST_F(ConvertConvertLikeTransformTests, ConvertLikeWithNodeOutput) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto data2 = std::make_shared<opset8::Parameter>(element::i8, Shape{1});
        auto constant = opset8::Constant::create(element::i8, Shape{}, {1});
        auto like = std::make_shared<opset8::Add>(data2, constant);
        auto cvtlike = std::make_shared<opset8::ConvertLike>(data, like);
        model = std::make_shared<ov::Model>(OutputVector{cvtlike}, ParameterVector{data, data2});
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto cvt = std::make_shared<opset8::Convert>(data, element::i8);
        model_ref = std::make_shared<ov::Model>(OutputVector{cvt}, ParameterVector{data});
    }
}

TEST_F(ConvertConvertLikeTransformTests, ConvertLikeWithDynamicType_Negative) {
    auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
    auto like = std::make_shared<opset8::Parameter>(element::dynamic, Shape{1});
    auto cvtlike = std::make_shared<opset8::ConvertLike>(data, like);
    model = std::make_shared<ov::Model>(OutputVector{cvtlike}, ParameterVector{data, like});
}
