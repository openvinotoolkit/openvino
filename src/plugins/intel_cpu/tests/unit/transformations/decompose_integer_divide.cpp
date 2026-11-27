// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1_decl.hpp>
#include <transformations/cpu_opset/common/pass/decompose_integer_divide.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/parameter.hpp"

using namespace testing;
using namespace ov::intel_cpu;

TEST_F(TransformationTestsF, DecomposeIntegerDivide_PythonDiv) {
    manager.register_pass<DecomposeIntegerDivide>();

    {
        auto lhs = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{4});
        auto rhs = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{4});
        auto divide = std::make_shared<ov::op::v1::Divide>(lhs, rhs, true);
        model = std::make_shared<ov::Model>(ov::OutputVector{divide}, ov::ParameterVector{lhs, rhs});
    }

    {
        auto lhs = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{4});
        auto rhs = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{4});
        auto divide = std::make_shared<ov::op::v1::Divide>(lhs, rhs, true);
        auto floor = std::make_shared<ov::opset1::Floor>(divide);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{floor}, ov::ParameterVector{lhs, rhs});
    }
}

TEST_F(TransformationTestsF, DecomposeIntegerDivide_TruncateDiv) {
    manager.register_pass<DecomposeIntegerDivide>();

    auto lhs = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{4});
    auto rhs = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{4});
    auto divide = std::make_shared<ov::op::v1::Divide>(lhs, rhs, false);
    model = std::make_shared<ov::Model>(ov::OutputVector{divide}, ov::ParameterVector{lhs, rhs});

    auto ref_lhs = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{4});
    auto ref_rhs = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{4});
    auto ref_divide = std::make_shared<ov::op::v1::Divide>(ref_lhs, ref_rhs, false);
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{ref_divide}, ov::ParameterVector{ref_lhs, ref_rhs});
}
