// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_mul_add_fq_block.hpp"

#include <memory>

#include "mul_add_fq_tail.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass::pattern;

ov::intel_cpu::ConvMulAddFQBlock::ConvMulAddFQBlock(const bool require_int_fq_output)
    : ov::pass::pattern::op::Block({}, {}, "ConvMulAddFQBlock") {
    auto u8_activation = any_input(type_matches(element::u8));
    auto u8_opt_convert = optional<ov::op::v0::Convert>({u8_activation});
    auto u8_zero_point = any_input();
    auto u8_opt_subtract = optional<ov::op::v1::Subtract>({u8_opt_convert, u8_zero_point});
    auto u8_weights = any_input(type_matches_any({element::i8, element::u8}));
    auto conv_u8 = wrap_type<ov::op::v1::Convolution>({u8_opt_subtract, u8_weights});

    auto i8_activation = any_input(type_matches(element::i8));
    auto i8_opt_convert = optional<ov::op::v0::Convert>({i8_activation});
    auto i8_zero_point = any_input();
    auto i8_opt_subtract = optional<ov::op::v1::Subtract>({i8_opt_convert, i8_zero_point});
    auto i8_weights = any_input(type_matches(element::i8));
    auto conv_i8 = wrap_type<ov::op::v1::Convolution>({i8_opt_subtract, i8_weights});

    auto conv = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{conv_u8, conv_i8});

    auto fake_quantize = append_mul_add_fq_tail(this, conv, require_int_fq_output);

    m_inputs = ov::OutputVector{conv};
    m_outputs = ov::OutputVector{fake_quantize};

    register_anchor("gemm", conv);
}
