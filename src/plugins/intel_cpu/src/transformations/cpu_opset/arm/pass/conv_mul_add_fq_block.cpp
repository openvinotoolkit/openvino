// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "conv_mul_add_fq_block.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/predicate.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass::pattern;

ov::intel_cpu::ConvMulAddFQBlock::ConvMulAddFQBlock(const bool require_int_fq_output)
    : ov::pass::pattern::op::Block({}, {}, "ConvMulAddFQBlock") {
    auto conv_i8_activation = any_input(type_matches(element::i8));
    auto conv_i8_weights = any_input(type_matches(element::i8));
    auto conv_i8 = wrap_type<ov::op::v1::Convolution>({conv_i8_activation, conv_i8_weights});

    auto conv_u8_activation = any_input(type_matches(element::u8));
    auto conv_i8_u8_weights = any_input(type_matches_any({element::i8, element::u8}));
    auto conv_u8 = wrap_type<ov::op::v1::Convolution>({conv_u8_activation, conv_i8_u8_weights});
    auto conv = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{conv_u8, conv_i8});

    auto multiply = wrap_type<ov::op::v1::Multiply>({conv, any_input()});
    auto bias_const = wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
        return !type_matches(ov::element::i32)(output);
    });
    auto add = wrap_type<ov::op::v1::Add>({multiply, bias_const});

    ov::pass::pattern::op::Predicate predicate =
        require_int_fq_output ? type_matches_any({element::i8, element::u8}) : ov::pass::pattern::op::Predicate();
    auto fake_quantize =
        wrap_type<ov::op::v0::FakeQuantize>({add, any_input(), any_input(), any_input(), any_input()}, predicate);

    m_inputs = ov::OutputVector{conv};
    m_outputs = ov::OutputVector{fake_quantize};

    register_anchor("convolution", conv);
    register_anchor("multiply", multiply);
    register_anchor("add", add);
    register_anchor("fake_quantize", fake_quantize);
}
