// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "matops_to_dwsc.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/cc/ngraph/itt.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

// Check if the previous node is convolution, groupconvolution or matmul followed by add
static bool IsFusable(const Output<Node>& parent) {
    if (nullptr == std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(parent.get_node()->shared_from_this()) ||
        nullptr == std::dynamic_pointer_cast<ngraph::opset1::Convolution>(parent.get_node()->shared_from_this()) ||
        nullptr == std::dynamic_pointer_cast<ngraph::opset1::MatMul>(parent.get_node()->shared_from_this()))
        return false;
    return true;
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

    uint64_t N, C, H, W;
    switch (input_shape.size()) {
    case 4:
        N = input_shape[0];
        C = input_shape[1];
        H = input_shape[2];
        W = input_shape[3];
        break;
    case 2:
        N = 1;
        C = input_shape[1];
        H = 1;
        W = input_shape[0];
        break;
    default:
        return false;
    }

    uint64_t N_params, C_params, H_params, W_params;
    switch (params_shape.size()) {
    case 4:
        N_params = params_shape[0];
        C_params = params_shape[1];
        H_params = input_shape[2];
        W_params = input_shape[3];
        break;
    case 2:
        N_params = params_shape[0];
        C_params = params_shape[1];
        H_params = 1;
        W_params = 1;
        break;
    case 1:
        N_params = 1;
        C_params = params_shape[0];
        H_params = 1;
        W_params = 1;
        break;
    default:
        return false;
    }

    // getting the parent(previous) node
    const Output<Node>& parent = math_node->input_value(0);
    if (nullptr != std::dynamic_pointer_cast<ngraph::opset1::Multiply>(math_node) && IsFusable(parent))
        return false;

    auto G = C;             // number of groups
    auto Co = (uint64_t)1;  // no. kernels in a group
    auto Ci = (uint64_t)1;  // no. input channels in a group
    auto Kh = (uint64_t)1;  // kernel height
    auto Kw = (uint64_t)1;  // kernel width

    // Initialize dwsc (group convolution) kernel weights as 1s
    std::vector<float> dwsc_weights(Kh * Kw * G, 0.0f);
    float* dwsc_weight_ptr = dwsc_weights.data();
    if (nullptr == std::dynamic_pointer_cast<ngraph::opset1::Multiply>(math_node)) {
        for (uint32_t i = 0; i < Kh * Kw * G; i++)
            *(dwsc_weight_ptr + i) = 1;
    } else {
        auto weights_const =
            std::dynamic_pointer_cast<ngraph::opset1::Constant>(math_node->input_value(1).get_node_shared_ptr());
        const float* weights_ptr = weights_const->get_data_ptr<float>();
        for (uint32_t i = 0; i < Kh * Kw * G; i++)
            *(dwsc_weight_ptr + i) = *(weights_ptr + i);
    }

    // Create a constant vector of weights
    auto dwsc_weights_const =
        ngraph::op::Constant::create(ngraph::element::f32, Shape{G, Co, Ci, Kh, Kw}, dwsc_weights);
    dwsc_weights_const->set_friendly_name("dwsc_weights");

    // Initializing group convolution parameters
    const Strides& strides = Strides({1, 1});
    const CoordinateDiff& pads_begin = CoordinateDiff({0, 0});
    const CoordinateDiff& pads_end = CoordinateDiff({0, 0});
    const Strides& dilations = Strides({1, 1});
    const ngraph::op::PadType& auto_pad = ngraph::op::PadType::EXPLICIT;

    // Initializing the dwsc node
    std::shared_ptr<ov::op::v1::GroupConvolution> new_dwsc;
    if (4 != input_shape.size()) {
        std::shared_ptr<ov::op::v1::Reshape> input_4d;
        if (W > 1) {
            auto n_elements = input_shape[0] * input_shape[1];
            if (n_elements < 8 || n_elements > 65528 || n_elements % 8 != 0)
                return false;
            auto new_transpose = std::make_shared<ngraph::op::Transpose>(
                math_node->input_value(0),
                ngraph::op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            input_4d = std::make_shared<ngraph::opset1::Reshape>(
                new_transpose->output(0),
                ngraph::op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                false);
        } else
            input_4d = std::make_shared<ngraph::opset1::Reshape>(
                math_node->input_value(0),
                ngraph::op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                false);
        new_dwsc = std::make_shared<ngraph::opset1::GroupConvolution>(input_4d->output(0),
                                                                      dwsc_weights_const->output(0),
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad);
    } else {
        new_dwsc = std::make_shared<ngraph::opset1::GroupConvolution>(math_node->input_value(0),
                                                                      dwsc_weights_const->output(0),
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad);
    }

    new_dwsc->set_friendly_name("replace_math_operation");

    std::shared_ptr<ov::Node> skip_node;
    if (nullptr == std::dynamic_pointer_cast<ngraph::opset1::Multiply>(math_node)) {
        // creating a bias node
        auto bias_const =
            std::dynamic_pointer_cast<ngraph::opset1::Constant>(math_node->input_value(1).get_node_shared_ptr());
        const float* bias_ptr = bias_const->get_data_ptr<float>();
        std::vector<float> new_bias(N_params * C_params * H_params * W_params, 0.0f);
        float* new_bias_ptr = new_bias.data();

        if (nullptr == std::dynamic_pointer_cast<ngraph::opset1::Subtract>(math_node)) {
            for (size_t i = 0; i < N_params * C_params * H_params * W_params; i++)
                *(new_bias_ptr + i) = *(bias_ptr + i);
        } else {
            for (size_t i = 0; i < N_params * C_params * H_params * W_params; i++)
                *(new_bias_ptr + i) = -1 * (*(bias_ptr + i));
        }
        auto new_bias_const =
            ngraph::op::Constant::create(ngraph::element::f32, Shape{N_params, C_params, H_params, W_params}, new_bias);

        // creating a new add node
        auto new_add =
            std::make_shared<ngraph::opset1::Add>(new_dwsc->output(0), new_bias_const->output(0), auto_broadcast);
        skip_node = new_add;
    } else {
        skip_node = new_dwsc;
    }

    // Reshape for different input dimensions
    if (4 == input_shape.size()) {
        ngraph::replace_node(math_node, skip_node);
    } else {
        std::shared_ptr<ngraph::opset1::Reshape> new_reshape;
        if (W > 1) {
            new_reshape =
                std::make_shared<ngraph::opset1::Reshape>(skip_node->output(0),
                                                          ngraph::op::Constant::create(ngraph::element::i64,
                                                                                       Shape{input_shape.size()},
                                                                                       {input_shape[1], input_shape[0]})
                                                              ->output(0),
                                                          false);
            auto new_shape = new_reshape->output(0).get_shape();
            auto n_elements = new_shape[0] * new_shape[1];
            if (n_elements < 8 || n_elements > 65528 || n_elements % 8 != 0)
                return false;

            auto untranspose = std::make_shared<ngraph::op::Transpose>(
                new_reshape->output(0),
                ngraph::op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            ngraph::replace_node(math_node, untranspose);
        } else {
            new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                skip_node->output(0),
                ngraph::op::Constant::create(ngraph::element::i64, Shape{input_shape.size()}, input_shape)->output(0),
                false);
            ngraph::replace_node(math_node, new_reshape);
        }
    }

    return true;
}

AddDecomposition::AddDecomposition() {
    MATCHER_SCOPE(AddDecomposition);
    auto add = ngraph::pattern::wrap_type<ngraph::opset1::Add>();

    ov::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto add = std::dynamic_pointer_cast<ngraph::opset1::Add>(m.get_match_root());

        return Decompose(add);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, matcher_name);
    this->register_matcher(m, callback);
}

SubDecomposition::SubDecomposition() {
    MATCHER_SCOPE(SubDecomposition);
    auto sub = ngraph::pattern::wrap_type<ngraph::opset1::Subtract>();

    ov::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto sub = std::dynamic_pointer_cast<ngraph::opset1::Subtract>(m.get_match_root());

        return Decompose(sub);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sub, matcher_name);
    this->register_matcher(m, callback);
}

MulDecomposition::MulDecomposition() {
    MATCHER_SCOPE(MulDecomposition);
    auto mul = ngraph::pattern::wrap_type<ngraph::opset1::Multiply>();

    ov::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto mul = std::dynamic_pointer_cast<ngraph::opset1::Multiply>(m.get_match_root());

        return Decompose(mul);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov