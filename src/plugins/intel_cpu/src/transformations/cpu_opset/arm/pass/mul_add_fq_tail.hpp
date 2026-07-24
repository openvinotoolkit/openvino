// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/predicate.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

/*
 * Description:
 *     Shared tail of the ARM int8 requantization pattern blocks ConvMulAddFQBlock and FCMulAddFQBlock.
 *     Both blocks share the identical GEMM -> Multiply -> Add(non-i32 constant bias) -> FakeQuantize
 */

namespace ov::intel_cpu {

inline ov::Output<ov::Node> append_mul_add_fq_tail(ov::pass::pattern::op::Block* block,
                                                   const ov::Output<ov::Node>& gemm,
                                                   bool require_int_fq_output) {
    using namespace ov::pass::pattern;

    auto multiply = wrap_type<ov::op::v1::Multiply>({gemm, any_input()});
    auto bias_const = wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
        return !type_matches(ov::element::i32)(output);
    });
    auto add = wrap_type<ov::op::v1::Add>({multiply, bias_const});

    ov::pass::pattern::op::Predicate predicate = require_int_fq_output
                                                     ? type_matches_any({ov::element::i8, ov::element::u8})
                                                     : ov::pass::pattern::op::Predicate();
    auto fake_quantize =
        wrap_type<ov::op::v0::FakeQuantize>({add, any_input(), any_input(), any_input(), any_input()}, predicate);

    block->register_anchor("multiply", multiply);
    block->register_anchor("add", add);
    block->register_anchor("fake_quantize", fake_quantize);

    return fake_quantize;
}

}  // namespace ov::intel_cpu
