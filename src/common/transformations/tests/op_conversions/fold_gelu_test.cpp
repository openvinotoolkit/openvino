// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/fold_gelu.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::opset12;
using namespace testing;

TEST_F(TransformationTestsF, FoldGeluOperation) {
    {
        auto param = std::make_shared<Parameter>(element::f32, Shape{1006, 2, 100, 3, 4096});
        auto const1 = Constant::create(element::f32, Shape{1, 1, 1}, std::vector<float>{0.044715});

        auto mul1 = std::make_shared<Multiply>(param, const1);
        auto mul2 = std::make_shared<Multiply>(mul1, param);

        auto const2 = Constant::create(element::f32, Shape{1, 1, 1}, std::vector<float>{1.0});
        auto add1 = std::make_shared<Add>(const2, mul2);

        auto const3 = Constant::create(element::f32, Shape{1, 1, 1}, std::vector<float>{0.7978845608028654});
        auto mul3 = std::make_shared<Multiply>(param, const3);

        auto mul4 = std::make_shared<Multiply>(add1, mul3);
        auto tan = std::make_shared<Tanh>(mul4);

        auto const4 = Constant::create(element::f32, Shape{1, 1, 1}, std::vector<float>{1.0});
        auto add2 = std::make_shared<Add>(tan, const4);

        auto const5 = Constant::create(element::f32, Shape{1, 1, 1}, std::vector<float>{0.5});
        auto mul5 = std::make_shared<Multiply>(param, const5);

        auto mul6 = std::make_shared<Multiply>(add2, mul5);

        auto result = std::make_shared<Result>(mul6);
        model = std::make_shared<Model>(NodeVector{result}, ParameterVector{param});

        manager.register_pass<ov::pass::FoldGelu>();
    }

    {
        auto param = std::make_shared<Parameter>(element::f32, Shape{1006, 2, 100, 3, 4096});
        auto gelu = std::make_shared<Gelu>(param, GeluApproximationMode::TANH);
        auto result = std::make_shared<Result>(gelu);
        model_ref = std::make_shared<Model>(NodeVector{result}, ParameterVector{param});
    }
}
