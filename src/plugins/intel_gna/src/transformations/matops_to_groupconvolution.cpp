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

// Check if the parent node is convolution, groupconvolution or matmul (add can be fused)
static bool IsFuseable(std::shared_ptr<ov::Node> node) {
    const Output<Node>& parent = node->input_value(0);

    if (nullptr != std::dynamic_pointer_cast<ngraph::opset8::GroupConvolution>(parent.get_node()->shared_from_this()) ||
        nullptr != std::dynamic_pointer_cast<ngraph::opset8::Convolution>(parent.get_node()->shared_from_this()) ||
        nullptr != std::dynamic_pointer_cast<ngraph::opset8::MatMul>(parent.get_node()->shared_from_this()))
        return true;
    return false;
}

static std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetConvParams(bool isMathOpOnFirstDim,
                                                                        ov::Shape input_shape) {
    if (input_shape.size() == 3 && input_shape.front() == 1) {
        input_shape.erase(std::begin(input_shape));
    }

    if (input_shape.size() == 4) {
        return std::make_tuple(input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    } else if (input_shape.size() == 3) {
        // TODO: check
        return std::make_tuple(input_shape[0], 1, input_shape[1], input_shape[2]);
    } else if (input_shape.size() == 2) {
        return std::make_tuple(1,
                               (isMathOpOnFirstDim) ? input_shape[0] : input_shape[1],
                               1,
                               (isMathOpOnFirstDim) ? input_shape[1] : input_shape[0]);
    } else {
        return std::make_tuple(1, input_shape[0], 1, 1);
    }
}
static std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetBiasParams(bool isMathOpOnFirstDim,
                                                                        ov::Shape params_shape) {
    if (params_shape.size() == 3 && params_shape.front() == 1) {
        params_shape.erase(std::begin(params_shape));
    }

    if (params_shape.size() == 4) {
        return std::make_tuple(params_shape[0], params_shape[1], params_shape[2], params_shape[3]);
    } else if (params_shape.size() == 3) {
        // TODO: Check
        return std::make_tuple(params_shape[0], 1, params_shape[1], params_shape[2]);
    } else if (params_shape.size() == 2) {
        return std::make_tuple((isMathOpOnFirstDim) ? params_shape[1] : params_shape[0],
                               (isMathOpOnFirstDim) ? params_shape[0] : params_shape[1],
                               1,
                               1);
    } else {
        return std::make_tuple(1, params_shape[0], 1, 1);
    }
}

static bool TransposeValidation(ov::Shape input_shape) {
    auto n_elements = input_shape[0] * input_shape[1];
    if (n_elements < limitations::affineMaxBatchSize || n_elements > limitations::bufferMaxSize ||
        n_elements % limitations::noOfInputsDivisor != 0)
        return false;
    return true;
}

std::shared_ptr<ov::Node> createReshapeNode(std::shared_ptr<ov::Node> node, uint32_t size, ov::Shape shape) {
    auto reshape_node = std::make_shared<ngraph::opset8::Reshape>(
        node->output(0),
        ngraph::op::Constant::create(ngraph::element::i64, Shape{size}, shape)->output(0),
        false);

    return reshape_node;
}

std::shared_ptr<ov::Node> createBiasConst(std::shared_ptr<ov::Node> math_node,
                                          bool isMathOpOnFirstDim,
                                          ov::Shape params_shape) {
    uint32_t N_params, C_params, H_params, W_params = 0;
    std::tie(N_params, C_params, H_params, W_params) = GetBiasParams(isMathOpOnFirstDim, params_shape);

    auto bias_const =
        std::dynamic_pointer_cast<ngraph::opset8::Constant>(math_node->input_value(1).get_node_shared_ptr());

    const float* bias_ptr = bias_const->get_data_ptr<float>();

    std::vector<float> new_bias(N_params * C_params * H_params * W_params, 0.0f);
    new_bias.assign(bias_ptr, bias_ptr + N_params * C_params * H_params * W_params);
    if (nullptr != std::dynamic_pointer_cast<ngraph::opset8::Subtract>(math_node)) {
        std::transform(new_bias.cbegin(), new_bias.cend(), new_bias.begin(), std::negate<float>());
    }

    auto new_bias_const =
        ngraph::op::Constant::create(ngraph::element::f32, Shape{N_params, C_params, H_params, W_params}, new_bias);

    return new_bias_const;
}

std::shared_ptr<ov::Node> createWeightConst(std::shared_ptr<ov::Node> math_node, uint32_t num_channels) {
    auto groups_num = num_channels;
    auto kernel_num_per_group = (uint64_t)1;
    auto input_channel_num_per_group = (uint64_t)1;
    auto kernel_height = (uint64_t)1;
    auto kernel_width = (uint64_t)1;

    std::vector<float> groupconv_weights(kernel_height * kernel_width * groups_num, 0.0f);
    if (nullptr == std::dynamic_pointer_cast<ngraph::opset8::Multiply>(math_node)) {
        std::fill(groupconv_weights.begin(), groupconv_weights.end(), 1.0f);
    } else {
        auto weights_const =
            std::dynamic_pointer_cast<ngraph::opset8::Constant>(math_node->input_value(1).get_node_shared_ptr());
        const float* weights_ptr = weights_const->get_data_ptr<float>();
        groupconv_weights.assign(weights_ptr, weights_ptr + kernel_height * kernel_width * groups_num);
    }

    auto groupconv_weights_const = ngraph::op::Constant::create(
        ngraph::element::f32,
        Shape{groups_num, kernel_num_per_group, input_channel_num_per_group, kernel_height, kernel_width},
        groupconv_weights);

    return groupconv_weights_const;
}

std::shared_ptr<ov::Node> createGroupConvolutionNode(std::shared_ptr<ov::Node> math_node,
                                                     bool isMathOpOnFirstDim,
                                                     ov::Shape input_shape) {
    uint32_t N, C, H, W = 0;
    std::tie(N, C, H, W) = GetConvParams(isMathOpOnFirstDim, input_shape);

    auto dwsc_weights_const = createWeightConst(math_node, C);
    dwsc_weights_const->set_friendly_name("dwsc_weights");

    // Initializing group convolution parameters
    const Strides& strides = Strides({1, 1});
    const CoordinateDiff& pads_begin = CoordinateDiff({0, 0});
    const CoordinateDiff& pads_end = CoordinateDiff({0, 0});
    const Strides& dilations = Strides({1, 1});
    const ngraph::op::PadType& auto_pad = ngraph::op::PadType::EXPLICIT;

    std::shared_ptr<ov::op::v1::GroupConvolution> new_dwsc;
    if (4 != input_shape.size()) {
        std::shared_ptr<ov::Node> input_4d;
        if ((W > 1) && (isMathOpOnFirstDim == false)) {
            TransposeValidation(input_shape);
            auto new_transpose = std::make_shared<ngraph::opset8::Transpose>(
                math_node->input_value(0),
                ngraph::op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));

            input_4d = createReshapeNode(new_transpose, 4, {N, C, H, W});

        } else
            input_4d = createReshapeNode(math_node, 4, {N, C, H, W});
        new_dwsc = std::make_shared<ngraph::opset8::GroupConvolution>(input_4d->output(0),
                                                                      dwsc_weights_const->output(0),
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad);
    } else {
        new_dwsc = std::make_shared<ngraph::opset8::GroupConvolution>(math_node->input_value(0),
                                                                      dwsc_weights_const->output(0),
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad);
    }

    return new_dwsc;
}

static bool Decompose(std::shared_ptr<ov::Node> math_node) {
    const Output<Node>& input = math_node->input_value(0);
    const Output<Node>& params = math_node->input_value(1);

    auto input_shape = input.get_shape();
    auto params_shape = params.get_shape();
    auto auto_broadcast = math_node->get_autob();
    auto output_shape = math_node->get_output_shape(0);

    if (input_shape.size() == 0)
        return false;

    // Check for 2D input on which dimension the math operation is carried out.
    bool isMathOpOnFirstDim = (input_shape.size() == 2) && ((input_shape.front() == params_shape.front()));

    if (nullptr != std::dynamic_pointer_cast<ngraph::opset8::Multiply>(math_node) && IsFuseable(math_node))
        return false;

    auto new_groupconv = createGroupConvolutionNode(math_node, isMathOpOnFirstDim, input_shape);
    new_groupconv->set_friendly_name("replace_math_operation");

    std::shared_ptr<ov::Node> skip_node;
    if (nullptr == std::dynamic_pointer_cast<ngraph::opset8::Multiply>(math_node)) {
        auto new_bias_const = createBiasConst(math_node, isMathOpOnFirstDim, params_shape);

        auto new_add =
            std::make_shared<ngraph::opset8::Add>(new_groupconv->output(0), new_bias_const->output(0), auto_broadcast);

        skip_node = new_add;
    } else {
        skip_node = new_groupconv;
    }

    if (4 == input_shape.size()) {
        ngraph::replace_node(math_node, skip_node);
    } else {
        // TODO:
        uint32_t N, C, H, W = 0;
        std::tie(N, C, H, W) = GetConvParams(isMathOpOnFirstDim, input_shape);

        if ((W > 1) && (isMathOpOnFirstDim == false)) {
            auto new_reshape = createReshapeNode(skip_node, input_shape.size(), {input_shape[1], input_shape[0]});

            TransposeValidation(new_reshape->output(0).get_shape());
            auto untranspose = std::make_shared<ngraph::op::Transpose>(
                new_reshape->output(0),
                ngraph::op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
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