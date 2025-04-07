// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/hsigmoid_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, HSigmoidDecompositionTest) {
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<opset5::HSigmoid>(input);

        model = std::make_shared<ov::Model>(NodeVector{hsigmoid}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidDecomposition>();
    }

    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic(1));
        auto add_constant = opset5::Constant::create(element::f32, Shape{}, {3.0});
        auto add = std::make_shared<opset5::Add>(input, add_constant);
        auto relu = std::make_shared<opset5::Relu>(add);
        auto min_constant = opset5::Constant::create(element::f32, Shape{}, {6.0});
        auto min = std::make_shared<opset5::Minimum>(relu, min_constant);
        auto mul_constant = opset5::Constant::create(element::f32, Shape{}, {(1.0 / 6.0)});  // const(1/6)
        auto mul = std::make_shared<opset5::Multiply>(min, mul_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});
    }
}
