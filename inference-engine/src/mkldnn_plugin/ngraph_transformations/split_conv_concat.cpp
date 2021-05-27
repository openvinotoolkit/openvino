// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split_conv_concat.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/validation_util.hpp>
#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::SplitConvConcatPattern, "SplitConvConcatPattern", 0);

bool isSplitConvConcatPattern(const std::shared_ptr<ngraph::Node> node) {
    auto split = std::dynamic_pointer_cast<ngraph::opset1::Split>(node);
    if (!split) {
        return false;
    }
    const auto axisNode = ngraph::get_constant_from_source(split->input_value(1));
    if (!axisNode) {
        return false;
    }
    auto axis = axisNode->cast_vector<int64_t>()[0];
    if (axis < 0) {
        axis += split->get_input_shape(0).size();
    }
    const auto splitOutputsNum = split->get_output_size();
    if (axis != 1 || splitOutputsNum < 2) {
        return false;
    }

    // get first convolution attribute, with which we will compare other convolutions
    auto splitOutputs = split->get_output_target_inputs(0);
    if (splitOutputs.size() != 1) {
        return false;
    }
    const auto convRef = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(splitOutputs.begin()->get_node()->shared_from_this());
    if (!convRef || !std::dynamic_pointer_cast<ngraph::opset1::Constant>(convRef->input_value(1).get_node_shared_ptr()) || convRef->get_output_size() != 1) {
        return false;
    }
    const auto stridesRef = convRef->get_strides();
    const auto padsBeginRef = convRef->get_pads_begin();
    const auto padsEndRef = convRef->get_pads_end();
    const auto dilationsRef = convRef->get_dilations();
    const auto autoPadRef = convRef->get_auto_pad();

    const auto convOutputs = convRef->get_output_target_inputs(0);
    const auto concatRef = std::dynamic_pointer_cast<ngraph::opset1::Concat>(convOutputs.begin()->get_node()->shared_from_this());
    if (!concatRef) {
        return false;
    }
    auto concatAxis = concatRef->get_axis();
    if (axis < 0) {
        concatAxis += concatRef->get_output_shape(0).size();
    }
    if (axis != 1) {
        return false;
    }

    for (size_t i = 1; i < splitOutputsNum; i++) {
        auto splitOutputs = split->get_output_target_inputs(i);
        if (splitOutputs.size() != 1) {
            return false;
        }
        const auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(splitOutputs.begin()->get_node()->shared_from_this());
        if (!conv || !std::dynamic_pointer_cast<ngraph::opset1::Constant>(conv->input_value(1).get_node_shared_ptr()) || conv->get_output_size() != 1) {
            return false;
        }
        if (conv->get_strides() != stridesRef || conv->get_pads_begin() != padsBeginRef || conv->get_pads_end() != padsEndRef ||
                conv->get_dilations() != dilationsRef || conv->get_auto_pad() != autoPadRef) {
            return false;
        }
        const auto convOutputs = conv->get_output_target_inputs(0);
        const auto concat = std::dynamic_pointer_cast<ngraph::opset1::Concat>(convOutputs.begin()->get_node()->shared_from_this());
        if (!concat || concatRef != concat) {
            return false;
        }
    }

    return true;
}

MKLDNNPlugin::SplitConvConcatPattern::SplitConvConcatPattern() {
    auto split = ngraph::pattern::wrap_type<ngraph::opset1::Split>();

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        if (!isSplitConvConcatPattern(m.get_match_root())) {
            return false;
        }
        const auto split = std::dynamic_pointer_cast<ngraph::opset1::Split>(m.get_match_root());
        ngraph::OutputVector weights;
        const auto numGroups = split->get_output_size();
        const auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(split->get_output_target_inputs(0).begin()->get_node()->shared_from_this());
        ngraph::NodeVector rtInfoToCopy;
        rtInfoToCopy.push_back(conv);
        for (size_t i = 0; i < numGroups; i++) {
            const auto conv = split->get_output_target_inputs(i).begin()->get_node()->shared_from_this();
            rtInfoToCopy.push_back(conv);
            weights.push_back(conv->input_value(1));
        }
        const auto newWeight = std::make_shared<ngraph::opset1::Concat>(weights, 0);
        auto weightShape = newWeight->get_output_shape(0);
        weightShape.insert(weightShape.begin(), numGroups);
        weightShape[1] /= numGroups;
        const auto groupConvWeight = ngraph::op::util::reshapeTo(newWeight, weightShape);
        const auto groupConv = std::make_shared<ngraph::opset1::GroupConvolution>(split->input_value(0), groupConvWeight, conv->get_strides(),
                                                                                  conv->get_pads_begin(), conv->get_pads_end(), conv->get_dilations(),
                                                                                  conv->get_auto_pad());

        const auto concat = std::dynamic_pointer_cast<ngraph::opset1::Concat>(conv->get_output_target_inputs(0).begin()->get_node()->shared_from_this());
        ngraph::copy_runtime_info(rtInfoToCopy, groupConv);
        ngraph::replace_node(concat, groupConv);
        groupConv->set_friendly_name(concat->get_friendly_name());

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(split, "SplitConvConcatPattern");
    this->register_matcher(m, callback);
}
