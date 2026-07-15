// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_mul_add_fq_block.hpp"

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/predicate.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass::pattern;

ov::intel_cpu::FCMulAddFQBlock::FCMulAddFQBlock(const bool require_int_fq_output)
    : ov::pass::pattern::op::Block({}, {}, "FCMulAddFQBlock") {
    // int8 FullyConnected is lowered to a MatMul with i8 activations and i8 weights.
    auto activation = any_input(type_matches_any({element::i8, element::u8}));
    auto weights = any_input(type_matches(element::i8));
    auto matmul = wrap_type<ov::op::v0::MatMul>({activation, weights});

    auto multiply = wrap_type<ov::op::v1::Multiply>({matmul, any_input()});
    auto bias_const = wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
        return !type_matches(ov::element::i32)(output);
    });
    auto add = wrap_type<ov::op::v1::Add>({multiply, bias_const});

    ov::pass::pattern::op::Predicate predicate =
        require_int_fq_output ? type_matches_any({element::i8, element::u8}) : ov::pass::pattern::op::Predicate();
    auto fake_quantize =
        wrap_type<ov::op::v0::FakeQuantize>({add, any_input(), any_input(), any_input(), any_input()}, predicate);

    m_inputs = ov::OutputVector{matmul};
    m_outputs = ov::OutputVector{fake_quantize};

    register_anchor("matmul", matmul);
    register_anchor("multiply", multiply);
    register_anchor("add", add);
    register_anchor("fake_quantize", fake_quantize);
}
