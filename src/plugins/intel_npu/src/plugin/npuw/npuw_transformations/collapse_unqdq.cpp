// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "collapse_unqdq.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"

namespace {

bool is_all_zero_constant(const ov::Output<ov::Node>& output) {
    const auto constant = ov::util::get_constant_from_source(output);
    if (constant == nullptr) {
        return false;
    }

    const auto values = constant->cast_vector<double>();
    return std::all_of(values.begin(), values.end(), [](double value) {
        return value == 0.0;
    });
}

bool is_pointwise_kernel_shape(const ov::Shape& shape) {
    return shape.size() == 4 && shape[2] == 1 && shape[3] == 1;
}

bool is_matmul_side_shape(const ov::Shape& shape) {
    return shape.size() == 4 && shape[1] == 1;
}

bool is_conv_activation_shape(const ov::Shape& shape) {
    return shape.size() == 4 && shape[2] == 1;
}

std::size_t element_count(const ov::Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>{});
}

std::shared_ptr<ov::op::v0::Constant> make_i32_shape_constant(const std::vector<int32_t>& dims) {
    return ov::op::v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
}

ov::Output<ov::Node> preserve_output_type(const ov::Output<ov::Node>& replacement,
                                          const ov::element::Type& expected_type,
                                          const std::string& friendly_name) {
    if (replacement.get_element_type() == expected_type) {
        return replacement;
    }

    auto convert = std::make_shared<ov::op::v0::Convert>(replacement, expected_type);
    convert->set_friendly_name(friendly_name);
    return convert;
}

bool has_permutation(const ov::Output<ov::Node>& output, const std::vector<int64_t>& expected) {
    const auto constant = ov::util::get_constant_from_source(output);
    if (constant == nullptr) {
        return false;
    }

    return constant->cast_vector<int64_t>() == expected;
}

ov::Output<ov::Node> peel_passthrough_nodes(ov::Output<ov::Node> output) {
    while (true) {
        if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(output.get_node_shared_ptr())) {
            output = convert->input_value(0);
            continue;
        }
        if (const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(output.get_node_shared_ptr())) {
            output = reshape->input_value(0);
            continue;
        }
        return output;
    }
}

bool rewrite_conv_to_matmul(const std::shared_ptr<ov::op::v1::Convolution>& convolution) {
    const auto transpose_in = ov::as_type_ptr<ov::op::v1::Transpose>(convolution->input_value(0).get_node_shared_ptr());
    const auto weight_multiply =
        ov::as_type_ptr<ov::op::v1::Multiply>(convolution->input_value(1).get_node_shared_ptr());
    if (weight_multiply == nullptr) {
        return false;
    }

    ov::Output<ov::Node> matmul_input;
    if (transpose_in != nullptr) {
        if (!has_permutation(transpose_in->input_value(1), {0, 3, 1, 2})) {
            return false;
        }
        matmul_input = transpose_in->input_value(0);
    } else {
        const auto& conv_input_shape = convolution->input_value(0).get_shape();
        if (!is_conv_activation_shape(conv_input_shape)) {
            return false;
        }
        matmul_input = std::make_shared<ov::op::v1::Transpose>(
            convolution->input_value(0),
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 2, 3, 1}));
    }

    if (!is_matmul_side_shape(matmul_input.get_shape())) {
        return false;
    }

    ov::Output<ov::Node> weight_value;
    ov::Output<ov::Node> weight_parameter_source;
    ov::Output<ov::Node> scale_source;

    const auto match_branch = [&](const ov::Output<ov::Node>& weight_branch,
                                  const ov::Output<ov::Node>& scale_branch) -> bool {
        const auto weight_source = peel_passthrough_nodes(weight_branch);
        if (!ov::is_type<ov::op::v0::Parameter>(weight_source.get_node_shared_ptr())) {
            return false;
        }

        const auto peeled_scale_source = peel_passthrough_nodes(scale_branch);
        const auto scale_base_node = peeled_scale_source.get_node_shared_ptr();
        if (!ov::is_type<ov::op::v0::Parameter>(scale_base_node) &&
            !ov::is_type<ov::op::v0::Constant>(scale_base_node)) {
            return false;
        }

        const auto& conv_weight_shape = weight_branch.get_shape();
        if (!is_pointwise_kernel_shape(conv_weight_shape) ||
            element_count(peeled_scale_source.get_shape()) != conv_weight_shape[0]) {
            return false;
        }

        weight_value = weight_branch;
        weight_parameter_source = weight_source;
        scale_source = peeled_scale_source;
        return true;
    };

    if (!match_branch(weight_multiply->input_value(0), weight_multiply->input_value(1)) &&
        !match_branch(weight_multiply->input_value(1), weight_multiply->input_value(0))) {
        return false;
    }

    const auto& conv_weight_shape = convolution->input_value(1).get_shape();

    if (convolution->get_strides() != ov::Strides{1, 1} || convolution->get_dilations() != ov::Strides{1, 1} ||
        convolution->get_pads_begin() != ov::CoordinateDiff{0, 0} ||
        convolution->get_pads_end() != ov::CoordinateDiff{0, 0}) {
        return false;
    }

    auto weight_reshape =
        std::make_shared<ov::op::v1::Reshape>(weight_parameter_source,
                                              make_i32_shape_constant({static_cast<int32_t>(conv_weight_shape[0]),
                                                                       static_cast<int32_t>(conv_weight_shape[1])}),
                                              false);
    ov::Output<ov::Node> converted_weight = weight_reshape;
    if (weight_reshape->get_output_element_type(0) != weight_value.get_element_type()) {
        converted_weight = std::make_shared<ov::op::v0::Convert>(weight_reshape, weight_value.get_element_type());
    }

    auto matmul = std::make_shared<ov::op::v0::MatMul>(matmul_input, converted_weight, false, true);

    auto scale_reshape = std::make_shared<ov::op::v1::Reshape>(
        scale_source,
        make_i32_shape_constant({1, 1, 1, static_cast<int32_t>(conv_weight_shape[0])}),
        false);
    ov::Output<ov::Node> scaled_coeff = scale_reshape;
    if (scale_reshape->get_output_element_type(0) != matmul->get_output_element_type(0)) {
        scaled_coeff = std::make_shared<ov::op::v0::Convert>(scale_reshape, matmul->get_output_element_type(0));
    }
    auto scaled_output = std::make_shared<ov::op::v1::Multiply>(matmul, scaled_coeff);

    const auto consumers = convolution->output(0).get_target_inputs();
    if (consumers.size() == 1) {
        const auto consumer = *consumers.begin();
        if (const auto transpose_out =
                ov::as_type_ptr<ov::op::v1::Transpose>(consumer.get_node()->shared_from_this())) {
            if (has_permutation(transpose_out->input_value(1), {0, 2, 3, 1}) &&
                is_matmul_side_shape(transpose_out->output(0).get_shape())) {
                ov::Output<ov::Node> replacement = preserve_output_type(scaled_output->output(0),
                                                                        transpose_out->get_output_element_type(0),
                                                                        transpose_out->get_friendly_name());
                replacement.get_node_shared_ptr()->set_friendly_name(transpose_out->get_friendly_name());
                ov::replace_node(transpose_out, ov::OutputVector{replacement});
                return true;
            }
        }
    }

    auto transpose_back = std::make_shared<ov::op::v1::Transpose>(
        scaled_output,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 3, 1, 2}));
    ov::Output<ov::Node> replacement =
        preserve_output_type(transpose_back->output(0), convolution->get_output_element_type(0), convolution->get_friendly_name());
    replacement.get_node_shared_ptr()->set_friendly_name(convolution->get_friendly_name());
    ov::replace_node(convolution, ov::OutputVector{replacement});
    return true;
}

}  // namespace

bool ov::npuw::CollapseUNQDQ::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool rewritten = false;

    for (const auto& node : model->get_ordered_ops()) {
        if (const auto convolution = ov::as_type_ptr<ov::op::v1::Convolution>(node)) {
            rewritten = rewrite_conv_to_matmul(convolution) || rewritten;
            continue;
        }

        if (const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(node)) {
            if (is_all_zero_constant(subtract->input_value(1))) {
                ov::replace_node(subtract, ov::OutputVector{subtract->input_value(0)});
                rewritten = true;
            }
            continue;
        }

        auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(node);
        if (multiply == nullptr) {
            continue;
        }

        std::shared_ptr<ov::op::v1::Subtract> subtract;
        ov::Output<ov::Node> scale_source;
        if (auto candidate = ov::as_type_ptr<ov::op::v1::Subtract>(multiply->input_value(0).get_node_shared_ptr())) {
            subtract = candidate;
            scale_source = multiply->input_value(1);
        } else if (auto candidate =
                       ov::as_type_ptr<ov::op::v1::Subtract>(multiply->input_value(1).get_node_shared_ptr())) {
            subtract = candidate;
            scale_source = multiply->input_value(0);
        } else {
            continue;
        }

        const auto dequantized_convert =
            ov::as_type_ptr<ov::op::v0::Convert>(subtract->input_value(0).get_node_shared_ptr());
        if (dequantized_convert == nullptr) {
            continue;
        }

        const auto quantized_convert =
            ov::as_type_ptr<ov::op::v0::Convert>(dequantized_convert->input_value(0).get_node_shared_ptr());
        if (quantized_convert == nullptr) {
            continue;
        }

        const auto fake_quantize =
            ov::as_type_ptr<ov::op::v0::FakeQuantize>(quantized_convert->input_value(0).get_node_shared_ptr());
        if (fake_quantize == nullptr) {
            continue;
        }

        ov::Output<ov::Node> replacement = fake_quantize->input_value(0);
        replacement =
            preserve_output_type(replacement, multiply->get_output_element_type(0), multiply->get_friendly_name());

        ov::replace_node(multiply, ov::OutputVector{replacement});
        rewritten = true;
    }

    return rewritten;
}
