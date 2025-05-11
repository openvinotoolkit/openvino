// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/reduce_l1_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/opsets/opset4_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, ReduceL1DecompositionTest) {
    {
        auto data = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(1));
        auto axes = std::make_shared<opset4::Parameter>(element::i32, Shape{1});
        auto reduce_l1 = std::make_shared<opset4::ReduceL1>(data, axes, true);

        model = std::make_shared<ov::Model>(OutputVector{reduce_l1}, ParameterVector{data, axes});

        manager.register_pass<ov::pass::ReduceL1Decomposition>();
    }

    {
        auto data = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(1));
        auto axes = std::make_shared<opset4::Parameter>(element::i32, Shape{1});
        auto abs = std::make_shared<opset4::Abs>(data);
        auto reduce_l1 = std::make_shared<opset4::ReduceSum>(abs, axes, true);

        model_ref = std::make_shared<ov::Model>(OutputVector{reduce_l1}, ParameterVector{data, axes});
    }
}
