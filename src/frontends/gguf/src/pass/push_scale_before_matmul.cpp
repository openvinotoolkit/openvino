// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/push_scale_before_matmul.hpp"

#include <cmath>
#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
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

// The scalar constant behind `value`, looking through the precision conversions a translator puts in front
// of it (SCALE emits Multiply(x, ConvertLike(c, x))) and the shape-only ops that give it the rank of the tensor
// it scales (Unsqueeze of a scalar, as the residual multiplier is emitted). Null when `value` is anything else.
std::shared_ptr<ov::op::v0::Constant> scalar_constant(ov::Output<ov::Node> value) {
    const auto is_transparent = [](const ov::Node* node) {
        return ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::ConvertLike>(node) ||
               ov::is_type<ov::op::v0::Unsqueeze>(node) || ov::is_type<ov::op::v0::Squeeze>(node) ||
               ov::is_type<ov::op::v1::Reshape>(node);
    };
    while (is_transparent(value.get_node())) {
        value = value.get_node()->input_value(0);
    }
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(value.get_node_shared_ptr());
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
        if (!matmul) {
            return false;
        }
        const size_t scale_input = multiply->input_value(0).get_node() == matmul.get() ? 1 : 0;
        const auto scale_const = scalar_constant(multiply->input_value(scale_input));
        if (!scale_const) {
            return false;
        }
        const float scale = scale_const->cast_vector<float>()[0];
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
            return std::make_shared<ov::op::v1::Multiply>(
                value,
                ov::op::v0::Constant::create(value.get_element_type(), ov::Shape{}, {scale}));
        };

        ov::Output<ov::Node> new_activations;
        const auto product = ov::as_type_ptr<ov::op::v1::Multiply>(activations.get_node_shared_ptr());
        if (product && activations.get_target_inputs().size() == 1 &&
            !ov::is_type<ov::op::v0::Constant>(product->get_input_node_shared_ptr(0)) &&
            !ov::is_type<ov::op::v0::Constant>(product->get_input_node_shared_ptr(1))) {
            // Gated product: scale the factor that is not the activation function's output.
            const size_t scaled_input = ov::is_type<ov::op::v4::Swish>(product->get_input_node_shared_ptr(0)) ? 1 : 0;
            const size_t other_input = 1 - scaled_input;
            new_activations = std::make_shared<ov::op::v1::Multiply>(product->input_value(other_input),
                                                                     scaled(product->input_value(scaled_input)));
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
