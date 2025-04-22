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
        auto scalar = linear_ir->push_node<ov::snippets::op::Scalar>(input_precision, scalar_shape, 42.f);
        auto add1 = linear_ir->push_node<ov::opset10::Add>(scalar.second, scalar.second);
        auto add2 = linear_ir->push_node<ov::opset10::Add>(add1.second, scalar.second);
        auto result = linear_ir->push_node<ov::opset10::Result>(add2.second);
    }
    pipeline.register_pass<MoveScalarToConsumer>();
    {
        // No changes in IR are expected
        auto scalar = linear_ir_ref->push_node<ov::snippets::op::Scalar>(input_precision, scalar_shape, 42.f);
        auto add1 = linear_ir_ref->push_node<ov::opset10::Add>(scalar.second, scalar.second);
        auto add2 = linear_ir_ref->push_node<ov::opset10::Add>(add1.second, scalar.second);
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(add2.second);
    }
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
