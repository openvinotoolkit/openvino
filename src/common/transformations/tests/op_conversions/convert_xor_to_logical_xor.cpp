// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_xor_to_logical_xor.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertXorToLogicalXor) {
    {
        constexpr int N = 1;
        constexpr int C = 3;
        constexpr int H = 5;
        constexpr int W = 5;

        const auto data_shape = Shape{N, C, H, W};
        const auto input1 = std::make_shared<opset1::Parameter>(element::boolean, data_shape);
        const auto input2 = std::make_shared<opset1::Parameter>(element::boolean, data_shape);

        auto xor_op =
            std::make_shared<opset1::Xor>(input1, input2, ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY));

        model = std::make_shared<ov::Model>(NodeVector{xor_op}, ParameterVector{input1, input2});
        manager.register_pass<ov::pass::ConvertXorToLogicalXor>();
    }

    {
        constexpr int N = 1;
        constexpr int C = 3;
        constexpr int H = 5;
        constexpr int W = 5;

        const auto data_shape = Shape{N, C, H, W};
        const auto input1 = std::make_shared<opset10::Parameter>(element::boolean, data_shape);
        const auto input2 = std::make_shared<opset10::Parameter>(element::boolean, data_shape);

        auto logical_xor =
            std::make_shared<opset10::LogicalXor>(input1,
                                                  input2,
                                                  ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY));

        model_ref = std::make_shared<ov::Model>(NodeVector{logical_xor}, ParameterVector{input1, input2});
    }
}
