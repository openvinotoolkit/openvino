// Copyright (C) 2020-2026 Intel Corporation
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
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass;

ov::intel_cpu::pass::pattern::op::ConvMulAddFQBlock::ConvMulAddFQBlock(const bool require_int_fq_output)
    : Block({}, {}, "ConvMulAddFQBlock") {
    auto conv_i8_activation = ov::pass::pattern::any_input(ov::pass::pattern::type_matches(element::i8));
    auto conv_i8_weights = ov::pass::pattern::any_input(ov::pass::pattern::type_matches(element::i8));
    auto conv_i8 = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>({conv_i8_activation, conv_i8_weights});

    auto conv_u8_activation = ov::pass::pattern::any_input(ov::pass::pattern::type_matches(element::u8));
    auto conv_i8_u8_weights =
        ov::pass::pattern::any_input(ov::pass::pattern::type_matches_any({element::i8, element::u8}));
    auto conv_u8 =
        ov::pass::pattern::wrap_type<ov::op::v1::Convolution>({conv_u8_activation, conv_i8_u8_weights});
    auto conv = conv_u8 | conv_i8;

    auto multiply =
        ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({conv, ov::pass::pattern::any_input()});
    auto bias_const = ov::pass::pattern::wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
        return !ov::pass::pattern::type_matches(ov::element::i32)(output);
    });
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({multiply, bias_const});

    const auto fq_inputs = ov::OutputVector{add,
                                            ov::pass::pattern::any_input(),
                                            ov::pass::pattern::any_input(),
                                            ov::pass::pattern::any_input(),
                                            ov::pass::pattern::any_input()};
    std::shared_ptr<ov::Node> fake_quantize;
    if (require_int_fq_output) {
        fake_quantize = ov::pass::pattern::wrap_type<ov::op::v0::FakeQuantize>(
            fq_inputs,
            ov::pass::pattern::type_matches_any({element::i8, element::u8}));
    } else {
        fake_quantize = ov::pass::pattern::wrap_type<ov::op::v0::FakeQuantize>(fq_inputs);
    }

    m_inputs = ov::OutputVector{conv};
    m_outputs = ov::OutputVector{fake_quantize};

    register_anchor("convolution", conv);
    register_anchor("multiply", multiply);
    register_anchor("add", add);
    register_anchor("fake_quantize", fake_quantize);
}
