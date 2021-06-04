// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_power_static.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "op/power_static.hpp"
#include "op/fully_connected.hpp"
#include "utils/general_utils.h"

int getConstPort(const std::shared_ptr<ngraph::Node> &node) {
    const auto const1 = std::dynamic_pointer_cast<ngraph::opset1::Constant>(node->get_input_node_shared_ptr(0));
    const auto const2 = std::dynamic_pointer_cast<ngraph::opset1::Constant>(node->get_input_node_shared_ptr(1));
    int constPort = -1;
    if (const2) {
        constPort = 1;
    } else if (const1) {
        constPort = 0;
    }
    return constPort;
}

template <class BaseOp>
bool isConvertableToPowerStatic(const std::shared_ptr<BaseOp> &node) {
    const int constPort = getConstPort(node);
    if ((!node->get_input_element_type(0).is_real() && !node->get_input_element_type(1).is_real()) || !node->get_output_element_type(0).is_real() ||
            constPort == -1) {
        return false;
    }

    const int nonConstPort = 1 - constPort;
    auto input_rank = node->get_input_partial_shape(nonConstPort).rank();
    if (input_rank.is_dynamic())
        return false;
    auto const_shape = node->get_input_shape(constPort);
    return ngraph::shape_size(const_shape) == 1 &&
           input_rank.get_length() >= const_shape.size() &&
           !MKLDNNPlugin::one_of_castable(node->get_input_node_shared_ptr(nonConstPort)->get_type_info(), ngraph::opset1::NormalizeL2::type_info,
                                                                                                 ngraph::opset4::Interpolate::type_info,
                                                                                                 ngraph::opset1::Convolution::type_info,
                                                                                                 ngraph::opset1::GroupConvolution::type_info,
                                                                                                 ngraph::opset1::ConvolutionBackpropData::type_info,
                                                                                                 ngraph::opset1::GroupConvolutionBackpropData::type_info,
                                                                                                 MKLDNNPlugin::FullyConnectedNode::type_info,
                                                                                                 ngraph::op::v0::MVN::type_info,
                                                                                                 ngraph::opset6::MVN::type_info);
}

template <>
bool isConvertableToPowerStatic(const std::shared_ptr<ngraph::opset1::Power> &node) {
    auto input_rank = node->get_input_partial_shape(0).rank();
    auto const_shape = node->get_input_shape(1);
    if (input_rank.is_dynamic())
        return false;
    return std::dynamic_pointer_cast<ngraph::opset1::Constant>(node->get_input_node_shared_ptr(1)) != nullptr &&
           input_rank.get_length() >= const_shape.size() && ngraph::shape_size(const_shape) == 1;
}

template <class BaseOp>
std::shared_ptr<ngraph::Node> convert(const std::shared_ptr<BaseOp> &node) {
    const int constPort = getConstPort(node);
    const int nonConstPort = 1 - constPort;
    std::shared_ptr<ngraph::opset1::Constant> powerNode = std::dynamic_pointer_cast<ngraph::opset1::Constant>(node->get_input_node_shared_ptr(constPort));
    const float value = powerNode->cast_vector<float>()[0];
    if (std::is_same<BaseOp, ngraph::opset1::Power>::value) {
        return std::make_shared<MKLDNNPlugin::PowerStaticNode>(node->input(nonConstPort).get_source_output(), value, 1.0f, 0.0f,
                                                               node->output(0).get_element_type());
    } else if (std::is_same<BaseOp, ngraph::opset1::Add>::value) {
        return std::make_shared<MKLDNNPlugin::PowerStaticNode>(node->input(nonConstPort).get_source_output(), 1.0f, 1.0f, value,
                                                               node->output(0).get_element_type());
    } else if (std::is_same<BaseOp, ngraph::opset1::Subtract>::value) {
        float scale = 1.0f;
        float shift = value;
        if (constPort == 0) {
            scale *= -1.0f;
        } else {
            shift *= -1.0f;
        }
        return std::make_shared<MKLDNNPlugin::PowerStaticNode>(node->input(nonConstPort).get_source_output(), 1.0f, scale, shift,
                                                               node->output(0).get_element_type());
    } else if (std::is_same<BaseOp, ngraph::opset1::Multiply>::value) {
        return std::make_shared<MKLDNNPlugin::PowerStaticNode>(node->input(nonConstPort).get_source_output(), 1.f, value, 0.0f,
                                                               node->output(0).get_element_type());
    } else {
        throw ngraph::ngraph_error("ConvertToPowerStatic: op type is not supported");
    }
}

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::ConvertToPowerStatic, "ConvertToPowerStatic", 0);

MKLDNNPlugin::ConvertToPowerStatic::ConvertToPowerStatic() {
    ngraph::OutputVector twoInputs = {ngraph::pattern::any_input(ngraph::pattern::has_static_rank()),
                                      ngraph::pattern::any_input(ngraph::pattern::has_static_rank())};
    auto power = ngraph::pattern::wrap_type<ngraph::opset1::Power>(twoInputs);
    auto add = ngraph::pattern::wrap_type<ngraph::opset1::Add>(twoInputs);
    auto sub = ngraph::pattern::wrap_type<ngraph::opset1::Subtract>(twoInputs);
    auto mult = ngraph::pattern::wrap_type<ngraph::opset1::Multiply>(twoInputs);
    const auto candidate = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{power, add, sub, mult});

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher &m) {
        auto node = m.get_match_root();

        std::shared_ptr<ngraph::Node> toReplace = node;
        if (auto power = std::dynamic_pointer_cast<ngraph::opset1::Power>(node)) {
            if (!isConvertableToPowerStatic(power))
                return false;
            toReplace = convert(power);
        } else if (auto add = std::dynamic_pointer_cast<ngraph::opset1::Add>(node)) {
            if (!isConvertableToPowerStatic(add))
                return false;
            toReplace = convert(add);
        } else if (auto sub = std::dynamic_pointer_cast<ngraph::opset1::Subtract>(node)) {
            if (!isConvertableToPowerStatic(sub))
                return false;
            toReplace = convert(sub);
        } else if (auto mult = std::dynamic_pointer_cast<ngraph::opset1::Multiply>(node)) {
            if (!isConvertableToPowerStatic(mult))
                return false;
            toReplace = convert(mult);
        } else {
            throw ngraph::ngraph_error("ConvertToPowerStatic: op type is not supported");
        }
        toReplace->set_friendly_name(node->get_friendly_name());
        ngraph::copy_runtime_info(node, toReplace);
        ngraph::replace_node(node, toReplace);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(candidate, "ConvertToPowerStatic");
    this->register_matcher(m, callback);
}
