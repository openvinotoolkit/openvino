// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/hswish_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, HSwishDecompositionTest) {
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hswish = std::make_shared<opset4::HSwish>(input);

        model = std::make_shared<ov::Model>(NodeVector{hswish}, ParameterVector{input});

        manager.register_pass<ov::pass::HSwishDecomposition>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset4::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset4::Add>(input, add_constant);
        auto relu = std::make_shared<opset4::Relu>(add);
        auto min_constant = opset4::Constant::create(element::f16, Shape{}, {6.0});
        auto min = std::make_shared<opset4::Minimum>(relu, min_constant);
        auto mul_first = std::make_shared<opset4::Multiply>(input, min);
        auto mul_constant = opset4::Constant::create(element::f16, Shape{}, {0.1666666716});
        auto mul_second = std::make_shared<opset4::Multiply>(mul_first, mul_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul_second}, ParameterVector{input});
    }
}
