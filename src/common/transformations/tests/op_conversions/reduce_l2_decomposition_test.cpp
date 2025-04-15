// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/reduce_l2_decomposition.hpp"

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

TEST_F(TransformationTestsF, ReduceL2DecompositionTest) {
    {
        auto data = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(1));
        auto axes = std::make_shared<opset4::Parameter>(element::i32, Shape{1});
        auto reduce_l2 = std::make_shared<opset4::ReduceL2>(data, axes, true);

        model = std::make_shared<ov::Model>(NodeVector{reduce_l2}, ParameterVector{data, axes});
        manager.register_pass<ov::pass::ReduceL2Decomposition>();
    }

    {
        auto data = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(1));
        auto axes = std::make_shared<opset4::Parameter>(element::i32, Shape{1});
        auto pow = std::make_shared<opset4::Power>(data, opset4::Constant::create(element::f32, Shape{}, {2.0}));
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes, true);
        auto sqrt = std::make_shared<opset4::Sqrt>(reduce_sum);

        model_ref = std::make_shared<ov::Model>(NodeVector{sqrt}, ParameterVector{data, axes});
    }
}
