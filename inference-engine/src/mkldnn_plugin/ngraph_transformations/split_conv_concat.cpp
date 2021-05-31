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
    auto concat = std::dynamic_pointer_cast<ngraph::opset1::Concat>(node);
    if (!concat) {
        return false;
    }
    const auto concatAxis = ngraph::normalize_axis(concat.get(), concat->get_axis(), concat->get_output_shape(0).size());
    const auto concatInputsNum = concat->get_input_size();
    if (concatAxis != 1 || concatInputsNum < 2) {
        return false;
    }

    // get first convolution attribute, with which we will compare other convolutions
    const auto convRef = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(concat->get_input_node_shared_ptr(0));
    if (!convRef || !std::dynamic_pointer_cast<ngraph::opset1::Constant>(convRef->get_input_node_shared_ptr(1)) || convRef->get_output_size() != 1) {
        return false;
    }
    const auto stridesRef = convRef->get_strides();
    const auto padsBeginRef = convRef->get_pads_begin();
    const auto padsEndRef = convRef->get_pads_end();
    const auto dilationsRef = convRef->get_dilations();
    const auto autoPadRef = convRef->get_auto_pad();

    const auto splitRef = std::dynamic_pointer_cast<ngraph::opset1::Split>(convRef->get_input_node_shared_ptr(0));
    if (!splitRef) {
        return false;
    }
    const auto axisNode = ngraph::get_constant_from_source(splitRef->input_value(1));
    if (!axisNode) {
        return false;
    }
    auto splitAxis = ngraph::normalize_axis(splitRef.get(), axisNode->cast_vector<int64_t>()[0], splitRef->get_input_shape(0).size());
    if (splitAxis != 1) {
        return false;
    }

    for (size_t i = 1; i < concatInputsNum; i++) {
        const auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(concat->get_input_node_shared_ptr(i));
        if (!conv || !std::dynamic_pointer_cast<ngraph::opset1::Constant>(conv->get_input_node_shared_ptr(1)) || conv->get_output_size() != 1) {
            return false;
        }
        if (conv->get_strides() != stridesRef || conv->get_pads_begin() != padsBeginRef || conv->get_pads_end() != padsEndRef ||
                conv->get_dilations() != dilationsRef || conv->get_auto_pad() != autoPadRef) {
            return false;
        }
        const auto split = std::dynamic_pointer_cast<ngraph::opset1::Split>(conv->get_input_node_shared_ptr(0));
        if (!split || splitRef != split) {
            return false;
        }
    }

    return true;
}

MKLDNNPlugin::SplitConvConcatPattern::SplitConvConcatPattern() {
    auto concat = ngraph::pattern::wrap_type<ngraph::opset1::Concat>();

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        if (!isSplitConvConcatPattern(m.get_match_root())) {
            return false;
        }
        const auto concat = std::dynamic_pointer_cast<ngraph::opset1::Concat>(m.get_match_root());
        ngraph::OutputVector weights;
        const auto numGroups = concat->get_input_size();
        ngraph::NodeVector rtInfoToCopy;
        for (size_t i = 0; i < numGroups; i++) {
            const auto conv = concat->get_input_node_shared_ptr(i);
            rtInfoToCopy.push_back(conv);
            weights.push_back(conv->input_value(1));
        }
        const auto newWeight = std::make_shared<ngraph::opset1::Concat>(weights, 0);
        auto weightShape = newWeight->get_output_shape(0);
        weightShape.insert(weightShape.begin(), numGroups);
        weightShape[1] /= numGroups;
        const auto groupConvWeight = ngraph::op::util::reshapeTo(newWeight, weightShape);
        const auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(concat->get_input_node_shared_ptr(0));
        const auto split = conv->get_input_node_shared_ptr(0);
        const auto groupConv = std::make_shared<ngraph::opset1::GroupConvolution>(split->input_value(0), groupConvWeight, conv->get_strides(),
                                                                                  conv->get_pads_begin(), conv->get_pads_end(), conv->get_dilations(),
                                                                                  conv->get_auto_pad());

        ngraph::copy_runtime_info(rtInfoToCopy, groupConv);
        ngraph::replace_node(concat, groupConv);
        groupConv->set_friendly_name(concat->get_friendly_name());

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat, "SplitConvConcatPattern");
    this->register_matcher(m, callback);
}
