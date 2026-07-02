// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_to_matmul.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

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

// Returns true if `output` is a chain of one or more Convert nodes whose final
// input is a Parameter.  Such a chain is the compressed-weight pattern:
//   Parameter (i4/i8/…) → Convert → [Convert …] → Multiply(scale) → Conv
// We require at least one Convert so we never reshape raw quantised data.
bool is_convert_param_chain(const ov::Output<ov::Node>& output) {
    auto cur = output;
    while (true) {
        const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(cur.get_node_shared_ptr());
        if (convert == nullptr) {
            return false;
        }
        if (ov::is_type<ov::op::v0::Parameter>(convert->input_value(0).get_node_shared_ptr())) {
            return true;
        }
        cur = convert->input_value(0);
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

    // weight_convert_output: output of the outermost Convert in the
    // Convert(…Convert(Parameter)…) chain — already a float tensor with shape
    // [OC, IC, 1, 1].  We add a Reshape *after* this node so that the
    // Parameter shape [OC, IC, 1, 1] is preserved in the graph.
    ov::Output<ov::Node> weight_convert_output;
    ov::Output<ov::Node> scale_source;

    const auto match_branch = [&](const ov::Output<ov::Node>& weight_branch,
                                  const ov::Output<ov::Node>& scale_branch) -> bool {
        // Weight must be a chain of Converts whose root is a Parameter.
        // Any intermediate Subtract (zero-point) node will cause the chain
        // check to fail and the match to be skipped.
        if (!is_convert_param_chain(weight_branch)) {
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

        weight_convert_output = weight_branch;
        scale_source = peeled_scale_source;
        return true;
    };

    if (!match_branch(weight_multiply->input_value(0), weight_multiply->input_value(1)) &&
        !match_branch(weight_multiply->input_value(1), weight_multiply->input_value(0))) {
        return false;
    }

    const auto& conv_weight_shape = weight_convert_output.get_shape();  // [OC, IC, 1, 1]

    if (convolution->get_strides() != ov::Strides{1, 1} || convolution->get_dilations() != ov::Strides{1, 1} ||
        convolution->get_pads_begin() != ov::CoordinateDiff{0, 0} ||
        convolution->get_pads_end() != ov::CoordinateDiff{0, 0}) {
        return false;
    }

    // Reshape the float weight tensor from [OC, IC, 1, 1] to [OC, IC].
    // The Reshape is inserted *after* the Convert chain so the Parameter node
    // retains its original shape and the runtime can bind tensors as-is.
    auto weight_reshape =
        std::make_shared<ov::op::v1::Reshape>(weight_convert_output,
                                              make_i32_shape_constant({static_cast<int32_t>(conv_weight_shape[0]),
                                                                       static_cast<int32_t>(conv_weight_shape[1])}),
                                              false);

    auto matmul = std::make_shared<ov::op::v0::MatMul>(matmul_input, weight_reshape, false, true);

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
    ov::Output<ov::Node> replacement = preserve_output_type(transpose_back->output(0),
                                                            convolution->get_output_element_type(0),
                                                            convolution->get_friendly_name());
    replacement.get_node_shared_ptr()->set_friendly_name(convolution->get_friendly_name());
    ov::replace_node(convolution, ov::OutputVector{replacement});
    return true;
}

}  // namespace

class ConvToMatMulMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::ConvToMatMulMatcher");

    ConvToMatMulMatcher() {
        auto convolution_with_output_transpose_pattern = opp::wrap_type<ov::op::v1::Convolution>();
        auto output_transpose_pattern =
            opp::wrap_type<ov::op::v1::Transpose>({convolution_with_output_transpose_pattern, opp::any_input()});

        ov::matcher_pass_callback output_transpose_callback = [](opp::Matcher& matcher) {
            const auto output_transpose = ov::as_type_ptr<ov::op::v1::Transpose>(matcher.get_match_root());
            if (output_transpose == nullptr) {
                return false;
            }

            const auto convolution =
                ov::as_type_ptr<ov::op::v1::Convolution>(output_transpose->input_value(0).get_node_shared_ptr());
            if (convolution == nullptr) {
                return false;
            }

            return rewrite_conv_to_matmul(convolution);
        };

        register_matcher(std::make_shared<opp::Matcher>(output_transpose_pattern, "ConvToMatMulOutputTransposeMatcher"),
                         output_transpose_callback);

        auto convolution_pattern = opp::wrap_type<ov::op::v1::Convolution>();

        ov::matcher_pass_callback convolution_callback = [](opp::Matcher& matcher) {
            const auto convolution = ov::as_type_ptr<ov::op::v1::Convolution>(matcher.get_match_root());
            if (convolution == nullptr) {
                return false;
            }

            return rewrite_conv_to_matmul(convolution);
        };

        register_matcher(std::make_shared<opp::Matcher>(convolution_pattern, "ConvToMatMulMatcher"),
                         convolution_callback);
    }
};

ov::npuw::ConvToMatMul::ConvToMatMul() {
    add_matcher<ConvToMatMulMatcher>();
}
