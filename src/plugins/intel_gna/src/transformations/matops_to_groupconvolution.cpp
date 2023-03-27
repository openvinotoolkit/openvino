// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/matops_to_groupconvolution.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/cc/ngraph/itt.hpp>

#include "backend/gna_limitations.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

// Check if the parent node is convolution, groupconvolution or matmul (add(bias) can be fused)
static bool IsFuseable(std::shared_ptr<ov::Node> node, const Output<Node>& input) {
    const Output<Node>& parent = node->input_value(0);
    auto bias_shape = input.get_shape();

    if (nullptr != std::dynamic_pointer_cast<ngraph::opset8::MatMul>(parent.get_node()->shared_from_this())) {
        if ((bias_shape.front() == 1) || (bias_shape.back() == 1))
            return true;
        else
            return false;
    }
    if (nullptr != std::dynamic_pointer_cast<ngraph::opset8::GroupConvolution>(parent.get_node()->shared_from_this()) ||
        nullptr != std::dynamic_pointer_cast<ngraph::opset8::Convolution>(parent.get_node()->shared_from_this()))
        if ((bias_shape[0] != 1) || (bias_shape[2] != 1) || (bias_shape[3] != 1))
            return false;
        return true;
    return false;
}

static bool checkInputShapesEq(std::shared_ptr<ov::Node> node) {
    if (node->input_value(0).get_shape() == node->input_value(1).get_shape())
        return true;
    return false;
}

// Check for 2D input on which dimension the math operation is carried out.
static bool isMathOpOnFirstDim(std::shared_ptr<ov::Node> node) {
    return ((node->input_value(0).get_shape().front() == node->input_value(1).get_shape().front()));
}

static std::tuple<bool, uint32_t, uint32_t, uint32_t, uint32_t> GetTensorParams(const Output<Node>& input,
                                                                                std::shared_ptr<ov::Node> node,
                                                                                bool isConvInput) {
    auto input_shape = input.get_shape();

    if (input_shape.size() == 3 && input_shape.front() == 1) {
        input_shape.erase(std::begin(input_shape));
    }

    if (input_shape.size() == 4) {
        return std::make_tuple(true, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    } else if (input_shape.size() == 3) {
        return std::make_tuple(false, 0, 0, 0, 0);
    } else if (input_shape.size() == 2) {
        if (checkInputShapesEq(node))
            return std::make_tuple(true, 1, input_shape[0] * input_shape[1], 1, 1);
        else {
            if (isConvInput) {
                if (isMathOpOnFirstDim(node))
                    return std::make_tuple(true, 1, input_shape[0], 1, input_shape[1]);
                else
                    return std::make_tuple(true, 1, input_shape[1], 1, input_shape[0]);
            } else {
                if (input_shape.front() == 1)
                    return std::make_tuple(true, 1, input_shape[1], 1, input_shape[0]);
                else
                    return std::make_tuple(true, 1, input_shape[0], 1, input_shape[1]);
            }
        }
    } else {
        return std::make_tuple(true, 1, input_shape[0], 1, 1);
    }
}

static bool TransposeValidation(ov::Shape input_shape) {
    auto n_elements = input_shape[0] * input_shape[1];
    if (n_elements < limitations::affineMaxBatchSize || n_elements > limitations::bufferMaxSize ||
        n_elements % limitations::noOfInputsDivisor != 0)
        return false;
    return true;
}

std::shared_ptr<ov::Node> createReshapeNode(const Output<Node>& input, uint32_t size, ov::Shape shape) {
    auto reshape_node = std::make_shared<ngraph::opset8::Reshape>(
        input,
        ngraph::op::Constant::create(ngraph::element::i64, Shape{size}, shape)->output(0),
        false);

    return reshape_node;
}

std::shared_ptr<ov::Node> createTransposeNode(const Output<Node>& input) {
    auto transpose_node = std::make_shared<ngraph::opset8::Transpose>(
        input,
        ngraph::op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));

    return transpose_node;
}

std::shared_ptr<ov::Node> createBias(std::shared_ptr<ov::Node> math_node, const Output<Node>& params) {
    bool supported;
    uint32_t N_params = 0, C_params = 0, H_params = 0, W_params = 0;
    std::tie(supported, N_params, C_params, H_params, W_params) = GetTensorParams(params, math_node, false);
    if (!supported)
        return nullptr;

    auto bias_const = std::dynamic_pointer_cast<ngraph::opset8::Constant>(params.get_node_shared_ptr());
    if (bias_const == nullptr) {
        auto new_reshape = std::make_shared<ngraph::opset8::Reshape>(
            params,
            ngraph::op::Constant::create(ngraph::element::i64, Shape{4}, {N_params, C_params, H_params, W_params})->output(0),
            false);
        return new_reshape;
    }

    const float* bias_ptr = bias_const->get_data_ptr<float>();
    std::vector<float> new_bias(N_params * C_params * H_params * W_params, 0.0f);
    new_bias.assign(bias_ptr, bias_ptr + N_params * C_params * H_params * W_params);

    auto new_bias_const =
        ngraph::op::Constant::create(ngraph::element::f32, Shape{N_params, C_params, H_params, W_params}, new_bias);

    return new_bias_const;
}

std::shared_ptr<ov::Node> createWeights(std::shared_ptr<ov::Node> math_node,
                                        const Output<Node>& params,
                                        uint32_t num_channels) {
    auto groups_num = num_channels;
    auto kernel_num_per_group = (uint64_t)1;
    auto input_channel_num_per_group = (uint64_t)1;
    auto kernel_height = (uint64_t)1;
    auto kernel_width = (uint64_t)1;

    std::vector<float> groupconv_weights(kernel_height * kernel_width * groups_num, 0.0f);
    if (nullptr == std::dynamic_pointer_cast<ngraph::opset8::Multiply>(math_node)) {
        std::fill(groupconv_weights.begin(), groupconv_weights.end(), 1.0f);
        if (nullptr != std::dynamic_pointer_cast<ngraph::opset8::Subtract>(math_node)) {
            std::transform(groupconv_weights.cbegin(),
                           groupconv_weights.cend(),
                           groupconv_weights.begin(),
                           std::negate<float>());
        }
    } else {
        auto weights_const = std::dynamic_pointer_cast<ngraph::opset8::Constant>(params.get_node_shared_ptr());
        if (weights_const != nullptr) {
            const float* weights_ptr = weights_const->get_data_ptr<float>();
            groupconv_weights.assign(weights_ptr, weights_ptr + kernel_height * kernel_width * groups_num);
        } else {
            auto reshape_weights = std::make_shared<ngraph::opset8::Reshape>(
                params,
                ngraph::op::Constant::create(
                    ngraph::element::i64,
                    Shape{5},
                    Shape{groups_num, kernel_num_per_group, input_channel_num_per_group, kernel_height, kernel_width})
                    ->output(0),
                false);
            return reshape_weights;
        }
    }

    auto groupconv_weights_const = ngraph::op::Constant::create(
        ngraph::element::f32,
        Shape{groups_num, kernel_num_per_group, input_channel_num_per_group, kernel_height, kernel_width},
        groupconv_weights);

    return groupconv_weights_const;
}

std::shared_ptr<ov::Node> createGroupConvolutionNode(const Output<Node>& input,
                                                     const Output<Node>& params,
                                                     std::shared_ptr<ov::Node> math_node) {
    auto input_shape = input.get_shape();
    bool supported;
    uint32_t N = 0, C = 0, H = 0, W = 0;
    std::tie(supported, N, C, H, W) = GetTensorParams(input, math_node, true);
    if (!supported)
        return nullptr;

    auto groupconv_weights = createWeights(math_node, params, C);
    groupconv_weights->set_friendly_name("groupconv_weights");

    // Initializing group convolution parameters
    const Strides& strides = Strides({1, 1});
    const CoordinateDiff& pads_begin = CoordinateDiff({0, 0});
    const CoordinateDiff& pads_end = CoordinateDiff({0, 0});
    const Strides& dilations = Strides({1, 1});
    const ngraph::op::PadType& auto_pad = ngraph::op::PadType::EXPLICIT;

    std::shared_ptr<ov::op::v1::GroupConvolution> new_group_conv;
    if (4 != input_shape.size()) {
        std::shared_ptr<ov::Node> input_4d;
        if ((!checkInputShapesEq(math_node)) && (!isMathOpOnFirstDim(math_node))) {
            TransposeValidation(input.get_shape());
            auto new_transpose = createTransposeNode(input);
            input_4d = createReshapeNode(new_transpose, 4, {N, C, H, W});
        } else {
            input_4d = createReshapeNode(input, 4, {N, C, H, W});
        }
        new_group_conv = std::make_shared<ngraph::opset8::GroupConvolution>(input_4d->output(0),
                                                                            groupconv_weights->output(0),
                                                                            strides,
                                                                            pads_begin,
                                                                            pads_end,
                                                                            dilations,
                                                                            auto_pad);
    } else {
        new_group_conv = std::make_shared<ngraph::opset8::GroupConvolution>(input,
                                                                            groupconv_weights->output(0),
                                                                            strides,
                                                                            pads_begin,
                                                                            pads_end,
                                                                            dilations,
                                                                            auto_pad);
    }

    return new_group_conv;
}

static bool Decompose(std::shared_ptr<ov::Node> math_node) {
    ov::Output<ov::Node> input = math_node->input_value(0);
    ov::Output<ov::Node> params = math_node->input_value(1);

    if ((input.get_shape().size() == 2) && (!checkInputShapesEq(math_node))) {
        if ((math_node->input_value(0).get_shape().front() == 1 || math_node->input_value(0).get_shape().back() == 1)) {
            input = math_node->input_value(1);
            params = math_node->input_value(0);
        }
    }

    auto input_shape = input.get_shape();
    auto params_shape = params.get_shape();

    if (input_shape.size() == 0)
        return false;

    if (nullptr == std::dynamic_pointer_cast<ngraph::opset8::Multiply>(math_node) && IsFuseable(math_node, params))
        return false;

    auto new_groupconv = createGroupConvolutionNode(input, params, math_node);
    if (new_groupconv == nullptr)
        return false;
    new_groupconv->set_friendly_name("replace_math_operation");

    std::shared_ptr<ov::Node> skip_node;
    if (nullptr == std::dynamic_pointer_cast<ngraph::opset8::Multiply>(math_node)) {
        auto new_bias = createBias(math_node, params);
        if (new_bias == nullptr)
            return false;

        auto new_add = std::make_shared<ngraph::opset8::Add>(new_groupconv->output(0), new_bias->output(0));

        skip_node = new_add;
    } else {
        skip_node = new_groupconv;
    }

    if (4 == input_shape.size()) {
        ngraph::replace_node(math_node, skip_node);
    } else {
        if ((!checkInputShapesEq(math_node)) && (!isMathOpOnFirstDim(math_node))) {
            auto new_reshape = createReshapeNode(skip_node, input_shape.size(), {input_shape[1], input_shape[0]});
            TransposeValidation(new_reshape->output(0).get_shape());
            auto untranspose = createTransposeNode(new_reshape->output(0));
            ngraph::replace_node(math_node, untranspose);

        } else {
            auto new_reshape = createReshapeNode(skip_node, input_shape.size(), input_shape);
            ngraph::replace_node(math_node, new_reshape);
        }
    }

    return true;
}

AddDecomposition::AddDecomposition() {
    MATCHER_SCOPE(AddDecomposition);
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>();

    ov::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto add = std::dynamic_pointer_cast<ngraph::opset8::Add>(m.get_match_root());
        return Decompose(add);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, matcher_name);
    this->register_matcher(m, callback);
}

SubDecomposition::SubDecomposition() {
    MATCHER_SCOPE(SubDecomposition);
    auto sub = ngraph::pattern::wrap_type<ngraph::opset8::Subtract>();

    ov::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto sub = std::dynamic_pointer_cast<ngraph::opset8::Subtract>(m.get_match_root());

        return Decompose(sub);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sub, matcher_name);
    this->register_matcher(m, callback);
}

MulDecomposition::MulDecomposition() {
    MATCHER_SCOPE(MulDecomposition);
    auto mul = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>();

    ov::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto mul = std::dynamic_pointer_cast<ngraph::opset8::Multiply>(m.get_match_root());

        return Decompose(mul);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov