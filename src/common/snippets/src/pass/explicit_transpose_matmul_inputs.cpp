// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/subgraph.hpp"

bool ov::snippets::pass::ExplicitTransposeMatMulInputs::are_weights_scalar(const std::shared_ptr<ov::Node>& node) {
    const auto inputs = node->inputs();
    return std::all_of(inputs.begin() + 1, inputs.end(),
                       [](const ov::Input<ov::Node>& in) {
                           return in.get_partial_shape().is_static() && ov::shape_size(in.get_shape()) == 1;
                       });
}

void ov::snippets::pass::ExplicitTransposeMatMulInputs::extract(const ov::Input<ov::Node>& input) {
    auto parent = input.get_source_output().get_node_shared_ptr();
    auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(parent);
    while (!transpose && !ov::is_type<ov::op::v0::Parameter>(parent)) {
        // We can set supported order and transposed_<a|b>=false only if ops have scalar shapes to avoid shape mismatching
        if (!are_weights_scalar(parent))
            break;

        parent = parent->get_input_node_shared_ptr(0);
        transpose = ov::as_type_ptr<ov::op::v1::Transpose>(parent);
    }

    // If there isn't another Transpose, need to create new Transpose
    if (transpose) {
        const auto transpose_pattern = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(transpose_pattern,
                        "ExplicitTransposeMatMulInputs expects existing Transpose with Constant order");

        auto transposed_order = transpose_pattern->cast_vector<int32_t>();
        OPENVINO_ASSERT(transposed_order.size() > 2, "Incorrect Transpose order for ExplicitTransposeMatMulInputs");
        std::swap(*transposed_order.rbegin(), *(transposed_order.rbegin() + 1));

        auto new_transpose_order = std::make_shared<ov::op::v0::Constant>(transpose_pattern->get_element_type(),
                                                                          ov::Shape{transposed_order.size()},
                                                                          transposed_order);
        new_transpose_order->set_friendly_name(transpose_pattern->get_friendly_name());
        ov::copy_runtime_info(transpose_pattern, new_transpose_order);
        transpose->set_argument(1, new_transpose_order);
        return;
    }

    // Create new Transpose before Parameter
    OPENVINO_ASSERT(ov::is_type<opset1::Parameter>(parent),
                    "ExplicitTransposeMatMulInputs expects Parameter in cases when there isn't existing Transpose on input");
    const auto& consumers = parent->get_output_target_inputs(0);
    OPENVINO_ASSERT(consumers.size() == 1,
                    "ExplicitTransposeMatMulInputs expects Parameter with one consumer in cases when there isn't existing Transpose on input");
    // Extract Transpose from MatMul
    OPENVINO_ASSERT(input.get_partial_shape().rank().is_static(), "ExplicitTransposeMatMulInputs supports only static ranks of shapes");

    const auto rank = input.get_partial_shape().size();
    std::vector<size_t> transpose_order(rank, 0);
    std::iota(transpose_order.begin(), transpose_order.end(), 0);
    std::swap(transpose_order[rank - 1], transpose_order[rank - 2]);

    const auto constant_order = std::make_shared<opset1::Constant>(ov::element::i32, ov::Shape{rank}, transpose_order);
    const auto new_transpose = std::make_shared<opset1::Transpose>(parent, constant_order); // parent is Parameter
    const auto consumer_input = *(consumers.begin());
    consumer_input.replace_source_output(new_transpose);
}

ov::snippets::pass::ExplicitTransposeMatMulInputs::ExplicitTransposeMatMulInputs() {
    MATCHER_SCOPE(ExplicitTransposeMatMulInputs);

    auto m_matmul0 = std::make_shared<ov::op::v0::MatMul>(ov::pass::pattern::any_input(), ov::pass::pattern::any_input());

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul0, matcher_name),
        [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::ExplicitTransposeMatMulInputs")
            auto root = m.get_match_root();
            bool rewritten = false;

            auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(root);
            if (!matmul)
                return false;

            if (matmul->get_transpose_a()) {
                extract(matmul->input(0));
                matmul->set_transpose_a(false);
                rewritten |= true;
            }

            if (matmul->get_transpose_b() && !transformation_callback(matmul)) {
                extract(matmul->input(1));
                matmul->set_transpose_b(false);
                rewritten |= true;
            }

            return rewritten;
        });
}
