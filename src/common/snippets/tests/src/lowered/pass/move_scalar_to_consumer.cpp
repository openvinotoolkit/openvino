// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lir_test_utils.hpp"

#include "openvino/opsets/opset10.hpp"
#include "snippets/lowered/pass/move_scalar_to_consumer.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/store.hpp"
#include "snippets/op/scalar.hpp"

namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;

TEST_F(LoweredPassTestsF, MoveScalarToConsumer) {
    const auto input_precision = ov::element::i8;
    const ov::Shape scalar_shape{1};
    {
        auto relu_scalar = linear_ir->push_node<ov::snippets::op::Scalar>(input_precision, scalar_shape, 0.f);
        auto add_scalar = linear_ir->push_node<ov::snippets::op::Scalar>(input_precision, scalar_shape, 42.f);
        auto relu = linear_ir->push_node<ov::opset10::Relu>(relu_scalar.second);
        auto add1 = linear_ir->push_node<ov::opset10::Add>(add_scalar.second, add_scalar.second);
        auto add2 = linear_ir->push_node<ov::opset10::Add>(add1.second, add_scalar.second);
        auto result = linear_ir->push_node<ov::opset10::Result>(add2.second);
    }
    pipeline.register_pass<MoveScalarToConsumer>();
    {
        auto relu_scalar = linear_ir_ref->push_node<ov::snippets::op::Scalar>(input_precision, scalar_shape, 0.f);
        auto relu = linear_ir_ref->push_node<ov::opset10::Relu>(relu_scalar.second);
        auto add_scalar = linear_ir_ref->push_node<ov::snippets::op::Scalar>(input_precision, scalar_shape, 42.f);
        auto add1 = linear_ir_ref->push_node<ov::opset10::Add>(add_scalar.second, add_scalar.second);
        auto add2 = linear_ir_ref->push_node<ov::opset10::Add>(add1.second, add_scalar.second);
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(add2.second);
    }
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
