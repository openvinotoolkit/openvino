// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/hsigmoid_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, HSigmoidDecompositionTest) {
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<op::v5::HSigmoid>(input);

        model = std::make_shared<ov::Model>(NodeVector{hsigmoid}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidDecomposition>();
    }

    {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
        auto add_constant = op::v0::Constant::create(element::f32, Shape{}, {3.0});
        auto add = std::make_shared<op::v1::Add>(input, add_constant);
        auto relu = std::make_shared<op::v0::Relu>(add);
        auto min_constant = op::v0::Constant::create(element::f32, Shape{}, {6.0});
        auto min = std::make_shared<op::v1::Minimum>(relu, min_constant);
        auto mul_constant = op::v0::Constant::create(element::f32, Shape{}, {(1.0 / 6.0)});  // const(1/6)
        auto mul = std::make_shared<op::v1::Multiply>(min, mul_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});
    }
}
