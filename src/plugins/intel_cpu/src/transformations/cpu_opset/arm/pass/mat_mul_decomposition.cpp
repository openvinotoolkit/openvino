// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "mat_mul_decomposition.hpp"

#include "ov_ops/type_relaxed.hpp"

#include "openvino/opsets/opset1.hpp"
#include "openvino/core/rt_info.hpp"

ov::intel_cpu::MatMulDecomposition::MatMulDecomposition() {
    auto matMul = ov::pass::pattern::wrap_type<opset1::MatMul>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto matMul = std::dynamic_pointer_cast<opset1::MatMul>(m.get_match_root());
        if (!matMul) {
            return false;
        }

        // TODO: is it possible to move to matcher?
        const auto in_type1 = matMul->get_input_element_type(0);
        const auto in_type2 = matMul->get_input_element_type(1);
        if ((in_type1 != element::i8) && (in_type1 != element::u8) && (in_type2 != element::u8) && (in_type2 != element::u8)) {
            return false;
        }

//        const std::shared_ptr<ov::op::v0::MatMul> newMatMul = std::make_shared<ov::op::TypeRelaxed<ov::opset1::MatMul>>(
//                std::vector<element::Type>({ deqPrecision, deqPrecision }), std::vector<element::Type>({ deqPrecision }),
//                ov::op::TemporaryReplaceOutputType(dequantization1.data, deqPrecision).get(),
//                ov::op::TemporaryReplaceOutputType(dequantization2.data, deqPrecision).get(),
//                matMul->get_transpose_a(),
//                matMul->get_transpose_b());
//        const auto newMatMul = std::make_shared<ov::op::TypeRelaxed<ov::opset1::MatMul>>(
//                parent,
//                std::make_shared<ov::opset1::Constant>(deqPrecision, ov::Shape({}), std::vector<float>({ dequantizationSub })),
//                matMul->get_transpose_a(),
//                matMul->get_transpose_b());

        const auto newMatMul = matMul->clone_with_new_inputs({matMul->get_input_source_output(0), matMul->get_input_source_output(1)});
        // TODO: output type is hardcoded
        newMatMul->set_output_type(0, element::i32, matMul->get_output_partial_shape(0));
        const auto convert = std::make_shared<ov::opset1::Convert>(newMatMul, element::f32);
        replace_node(matMul, convert);
        ov::copy_runtime_info(matMul, {newMatMul, convert});
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matMul, "MatMulDecomposition");
    register_matcher(m, callback);
}