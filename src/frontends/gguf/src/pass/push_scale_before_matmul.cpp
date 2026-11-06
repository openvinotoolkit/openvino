// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/push_scale_before_matmul.hpp"

#include <cmath>
#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::frontend::gguf::pass {

namespace {
constexpr float min_scale = 1.0f / 64.0f;

std::shared_ptr<ov::op::v0::Constant> fold_scale_expression(const ov::Output<ov::Node>& value) {
    const auto node = value.get_node_shared_ptr();
    const auto convert_like = ov::as_type_ptr<ov::op::v1::ConvertLike>(node);
    if (convert_like && value.get_element_type().is_static()) {
        const auto source = fold_scale_expression(convert_like->input_value(0));
        if (source) {
            return ov::util::constantfold_subgraph(
                std::make_shared<ov::op::v0::Convert>(source, value.get_element_type()));
        }
    } else if (ov::is_type<ov::op::v0::Convert>(node)) {
        const auto source = fold_scale_expression(node->input_value(0));
        if (source) {
            return ov::util::constantfold_subgraph(node->clone_with_new_inputs({source}));
        }
    } else if (ov::is_type_any_of<ov::op::v0::Unsqueeze, ov::op::v0::Squeeze, ov::op::v1::Reshape>(node)) {
        return fold_scale_expression(node->input_value(0));
    } else {
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
        return constant && ov::shape_size(constant->get_shape()) == 1 ? constant : nullptr;
    }
    return nullptr;
}

std::shared_ptr<ov::op::v0::Constant> scalar_constant(ov::Output<ov::Node> value) {
    auto constant = fold_scale_expression(value);
    if (!constant || ov::shape_size(constant->get_shape()) != 1 || !constant->get_element_type().is_real()) {
        return nullptr;
    }
    return constant;
}
}  // namespace

PushScaleBeforeMatMul::PushScaleBeforeMatMul() {
    using namespace ov::pass::pattern;
    auto matmul_pattern = wrap_type<ov::op::v0::MatMul>();
    auto multiply_pattern = wrap_type<ov::op::v1::Multiply>({matmul_pattern, any_input()});

    const auto callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(matmul_pattern).get_node_shared_ptr());
        auto multiply = pattern_map.at(multiply_pattern).get_node_shared_ptr();
        const size_t scale_input = multiply->input_value(0).get_node() == matmul.get() ? 1 : 0;
        const auto scale_const = scalar_constant(multiply->input_value(scale_input));
        if (!scale_const) {
            return false;
        }
        const auto scale_rank = multiply->input_value(scale_input).get_partial_shape().rank();
        const auto matmul_rank = matmul->get_output_partial_shape(0).rank();
        if (scale_rank.is_dynamic() || matmul_rank.is_dynamic() || scale_rank.get_length() > matmul_rank.get_length()) {
            return false;
        }
        const double scale = scale_const->cast_vector<double>()[0];
        if (!(std::abs(scale) >= min_scale && std::abs(scale) < 1.0f)) {
            return false;
        }
        // The MatMul result must only be used by this Multiply; the ConvertLike that takes the result's
        // precision as a reference (the scale's own conversion) does not use its values.
        for (const auto& consumer : matmul->output(0).get_target_inputs()) {
            const auto node = consumer.get_node();
            const bool is_scale = node == multiply.get();
            const bool is_type_reference = ov::is_type<ov::op::v1::ConvertLike>(node) && consumer.get_index() == 1;
            if (!is_scale && !is_type_reference) {
                return false;
            }
        }

        const auto activations = matmul->input_value(0);
        const auto scaled = [&](const ov::Output<ov::Node>& value) -> ov::Output<ov::Node> {
            auto constant = ov::op::v0::Constant::create(value.get_element_type(), ov::Shape{}, {scale});
            auto result = std::make_shared<ov::op::v1::Multiply>(value, constant);
            ov::copy_runtime_info(multiply, {constant, result});
            return result;
        };

        ov::Output<ov::Node> new_activations;
        const auto product = ov::as_type_ptr<ov::op::v1::Multiply>(activations.get_node_shared_ptr());
        if (product && activations.get_target_inputs().size() == 1 &&
            !ov::is_type<ov::op::v0::Constant>(product->get_input_node_shared_ptr(0)) &&
            !ov::is_type<ov::op::v0::Constant>(product->get_input_node_shared_ptr(1))) {
            // Gated product: scale the factor that is not the activation function's output.
            const size_t scaled_input = ov::is_type<ov::op::v4::Swish>(product->get_input_node_shared_ptr(0)) ? 1 : 0;
            auto inputs = product->input_values();
            inputs[scaled_input] = scaled(inputs[scaled_input]);
            new_activations = product->clone_with_new_inputs(inputs);
            ov::copy_runtime_info(product, new_activations.get_node_shared_ptr());
        } else {
            new_activations = scaled(activations);
        }

        auto new_matmul = std::make_shared<ov::op::v0::MatMul>(new_activations,
                                                               matmul->input_value(1),
                                                               matmul->get_transpose_a(),
                                                               matmul->get_transpose_b());
        ov::copy_runtime_info({matmul, multiply}, new_matmul);
        ov::replace_node_update_name(multiply, new_matmul);
        return true;
    };

    register_matcher(std::make_shared<Matcher>(multiply_pattern, "PushScaleBeforeMatMul"), callback);
}

}  // namespace ov::frontend::gguf::pass
