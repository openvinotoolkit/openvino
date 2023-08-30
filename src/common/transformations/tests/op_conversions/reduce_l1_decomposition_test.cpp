// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/reduce_l1_decomposition.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, ReduceL1DecompositionTest) {
    {
        auto data = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(1));
        auto axes = std::make_shared<opset4::Parameter>(element::i32, Shape{1});
        auto reduce_l1 = std::make_shared<opset4::ReduceL1>(data, axes, true);

        model = std::make_shared<ov::Model>(NodeVector{reduce_l1}, ParameterVector{data, axes});

        manager.register_pass<ov::pass::ReduceL1Decomposition>();
    }

    {
        auto data = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(1));
        auto axes = std::make_shared<opset4::Parameter>(element::i32, Shape{1});
        auto abs = std::make_shared<opset4::Abs>(data);
        auto reduce_l1 = std::make_shared<opset4::ReduceSum>(abs, axes, true);

        model_ref = std::make_shared<ov::Model>(NodeVector{reduce_l1}, ParameterVector{data, axes});
    }
}
