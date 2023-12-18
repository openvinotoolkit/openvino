// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset10.hpp>
#include <plugin/transformations/move_convert_after_gather.hpp>
#include <transformations/utils/utils.hpp>
#include <openvino/pass/manager.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, MoveConvertAfterGatherTest) {
    {
        auto weights = ov::opset10::Constant::create(ov::element::f16, {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
        auto convert = std::make_shared<ov::opset10::Convert>(weights, ov::element::f32);
        auto indices_input = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1, 1});
        auto axis_const = ov::opset10::Constant::create(ov::element::i32, {1}, {0});
        auto gather = std::make_shared<ov::opset10::Gather>(convert, indices_input, axis_const);
        auto mul_const = ov::opset10::Constant::create(ov::element::f32, ov::Shape{1}, {2.f});
        auto mul = std::make_shared<ov::opset10::Multiply>(mul_const, gather);

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{indices_input});
        manager.register_pass<MoveConvertAfterGather>();
    }
    {
        auto weights = ov::opset10::Constant::create(ov::element::f16, {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
        auto indices_input = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1, 1});
        auto axis_const = ov::opset10::Constant::create(ov::element::i32, {1}, {0});
        auto gather = std::make_shared<ov::opset10::Gather>(weights, indices_input, axis_const);
        auto convert = std::make_shared<ov::opset10::Convert>(gather, ov::element::f32);
        auto mul_const = ov::opset10::Constant::create(ov::element::f32, ov::Shape{1}, {2.f});
        auto mul = std::make_shared<ov::opset10::Multiply>(mul_const, convert);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{indices_input});
    }
}
