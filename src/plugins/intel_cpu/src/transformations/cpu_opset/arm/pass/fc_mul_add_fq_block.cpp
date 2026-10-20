// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_mul_add_fq_block.hpp"

#include "mul_add_fq_tail.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass::pattern;

ov::intel_cpu::FCMulAddFQBlock::FCMulAddFQBlock(const bool require_int_fq_output)
    : ov::pass::pattern::op::Block({}, {}, "FCMulAddFQBlock") {
    // int8 FullyConnected is lowered to a MatMul with i8 activations and i8 weights.
    auto activation = any_input(type_matches_any({element::i8, element::u8}));
    auto weights = any_input(type_matches(element::i8));
    auto matmul = wrap_type<ov::op::v0::MatMul>({activation, weights});

    auto fake_quantize = append_mul_add_fq_tail(this, matmul, require_int_fq_output);

    m_inputs = ov::OutputVector{matmul};
    m_outputs = ov::OutputVector{fake_quantize};

    register_anchor("gemm", matmul);
}
