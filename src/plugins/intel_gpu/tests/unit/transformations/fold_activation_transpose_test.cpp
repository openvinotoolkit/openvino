// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"

#include "plugin/transformations/fold_activation_transpose.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

// Valid case
TEST_F(TransformationTestsF, FoldActivationTranspose1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto order1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose1 = std::make_shared<ov::op::v1::Transpose>(input1, order1);
        auto swish = std::make_shared<ov::op::v4::Swish>(transpose1);
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto order2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose2 = std::make_shared<ov::op::v1::Transpose>(input2, order2);
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, transpose2);
        auto order3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
        auto transpose3 = std::make_shared<ov::op::v1::Transpose>(mul, order3);
        transpose3->set_friendly_name("Multiply_transpose");

        model = std::make_shared<ov::Model>(ov::OutputVector{transpose3}, ov::ParameterVector{input1, input2});
        manager.register_pass<FoldActivationTranspose>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto swish = std::make_shared<ov::op::v4::Swish>(input1);
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, input2);
        mul->set_friendly_name("Multiply_transpose");

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{mul}, ov::ParameterVector{input1, input2});
    }
}

// Mismatching output transpose order
TEST_F(TransformationTestsF, FoldActivationTranspose2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto order1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose1 = std::make_shared<ov::op::v1::Transpose>(input1, order1);
        auto swish = std::make_shared<ov::op::v4::Swish>(transpose1);
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto order2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose2 = std::make_shared<ov::op::v1::Transpose>(input2, order2);
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, transpose2);
        auto order3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
        auto transpose3 = std::make_shared<ov::op::v1::Transpose>(mul, order3);

        model = std::make_shared<ov::Model>(ov::OutputVector{transpose3}, ov::ParameterVector{input1, input2});
        manager.register_pass<FoldActivationTranspose>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto order1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose1 = std::make_shared<ov::op::v1::Transpose>(input1, order1);
        auto swish = std::make_shared<ov::op::v4::Swish>(transpose1);
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto order2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose2 = std::make_shared<ov::op::v1::Transpose>(input2, order2);
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, transpose2);
        auto order3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
        auto transpose3 = std::make_shared<ov::op::v1::Transpose>(mul, order3);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{transpose3}, ov::ParameterVector{input1, input2});
    }
}

// Mismatching input1 layout
TEST_F(TransformationTestsF, FoldActivationTranspose3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 7, 3});
        auto order1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto transpose1 = std::make_shared<ov::op::v1::Transpose>(input1, order1);
        auto swish = std::make_shared<ov::op::v4::Swish>(transpose1);
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto order2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose2 = std::make_shared<ov::op::v1::Transpose>(input2, order2);
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, transpose2);
        auto order3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
        auto transpose3 = std::make_shared<ov::op::v1::Transpose>(mul, order3);

        model = std::make_shared<ov::Model>(ov::OutputVector{transpose3}, ov::ParameterVector{input1, input2});
        manager.register_pass<FoldActivationTranspose>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 7, 3});
        auto order1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto transpose1 = std::make_shared<ov::op::v1::Transpose>(input1, order1);
        auto swish = std::make_shared<ov::op::v4::Swish>(transpose1);
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto order2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose2 = std::make_shared<ov::op::v1::Transpose>(input2, order2);
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, transpose2);
        auto order3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
        auto transpose3 = std::make_shared<ov::op::v1::Transpose>(mul, order3);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{transpose3}, ov::ParameterVector{input1, input2});
    }
}

// Mismatching input2 layout
TEST_F(TransformationTestsF, FoldActivationTranspose4) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto order1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose1 = std::make_shared<ov::op::v1::Transpose>(input1, order1);
        auto swish = std::make_shared<ov::op::v4::Swish>(transpose1);
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 7, 3});
        auto order2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto transpose2 = std::make_shared<ov::op::v1::Transpose>(input2, order2);
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, transpose2);
        auto order3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
        auto transpose3 = std::make_shared<ov::op::v1::Transpose>(mul, order3);

        model = std::make_shared<ov::Model>(ov::OutputVector{transpose3}, ov::ParameterVector{input1, input2});
        manager.register_pass<FoldActivationTranspose>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 3, 7});
        auto order1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose1 = std::make_shared<ov::op::v1::Transpose>(input1, order1);
        auto swish = std::make_shared<ov::op::v4::Swish>(transpose1);
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, 7, 3});
        auto order2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto transpose2 = std::make_shared<ov::op::v1::Transpose>(input2, order2);
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, transpose2);
        auto order3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
        auto transpose3 = std::make_shared<ov::op::v1::Transpose>(mul, order3);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{transpose3}, ov::ParameterVector{input1, input2});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
