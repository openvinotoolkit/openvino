// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/align_concat_quantization_parameters.hpp"

#include <algorithm>
#include <assert.h>
#include <memory>
#include <unordered_set>
#include <set>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/layer_transformation.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

AlignConcatQuantizationParamters::AlignConcatQuantizationParamters(LayerTransformation::Params params) : params(params) {
    //
}

bool ngraph::pass::low_precision::AlignConcatQuantizationParamters::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        // create new
        auto fakeQuantize = ngraph::as_type_ptr<opset1::FakeQuantize>(node);
        if (fakeQuantize != nullptr) {
            if (!QuantizationDetails::outputLayoutIsSupported(fakeQuantize) || !QuantizationDetails::isSupportedLevel(fakeQuantize->get_levels())) {
                // TODO: LPT: not implemented
                // should be handled: branch is not quantized: need tests
                auto& rtInfo = node->get_rt_info();
                const auto attribute = std::make_shared<::ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>>(
                    std::make_shared<IntervalsAlignmentAttribute>(0.f, 0.f, false));
                rtInfo[ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>::type_info.name] = attribute;
                continue;
            }
            // TODO: FakeQuantize validation is skipped: if FakeQuantize will be not handled then ignore it

            float lowInterval;
            float highInterval;
            FakeQuantizeDequantization dequantization;
            {
                const auto targetInputs = node->output(0).get_target_inputs();
                if (targetInputs.size() == 1ul) {
                    auto input = *targetInputs.begin();
                    dequantization = NetworkHelper::getDequantizationBelow(input.get_node()->shared_from_this());
                }
            }
            if (dequantization.empty()) {
                const std::vector<float> lowIntervals = as_type<opset1::Constant>(node->get_input_node_ptr(3))->cast_vector<float>();
                lowInterval = *std::min_element(lowIntervals.begin(), lowIntervals.end());

                const std::vector<float> highIntervals = as_type<opset1::Constant>(node->get_input_node_ptr(4))->cast_vector<float>();
                highInterval = *std::max_element(highIntervals.begin(), highIntervals.end());

            } else {
                {
                    auto multiplyResult = dequantization.multiplyConstant == nullptr ?
                        node->get_input_node_ptr(3)->shared_from_this() :
                        fold<opset1::Multiply>(
                            foldConvert(node->get_input_node_ptr(3)->shared_from_this(), params.deqPrecision),
                            dequantization.multiplyConstant);

                    auto multiplyResultConstant = as_type_ptr<opset1::Constant>(multiplyResult);
                    auto intervals = multiplyResultConstant->cast_vector<float>();
                    lowInterval = *std::min_element(intervals.begin(), intervals.end());
                }

                {
                    auto multiplyResult = dequantization.multiplyConstant == nullptr ?
                        node->get_input_node_ptr(4)->shared_from_this() :
                        fold<opset1::Multiply>(
                            foldConvert(node->get_input_node_ptr(4)->shared_from_this(), params.deqPrecision),
                            dequantization.multiplyConstant);

                    auto multiplyResultConstant = as_type_ptr<opset1::Constant>(multiplyResult);
                    auto intervals = multiplyResultConstant->cast_vector<float>();
                    highInterval = *std::max_element(intervals.begin(), intervals.end());
                }
            }

            //const float highInterval = *std::max_element(highIntervals.begin(), highIntervals.end());

            auto& rtInfo = node->get_rt_info();

            //const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(ngraph::as_type_ptr<opset1::FakeQuantize>(node));
            //const LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(quantizationDetails);

            const auto attribute = std::make_shared<::ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>>(
                std::make_shared<IntervalsAlignmentAttribute>(lowInterval, highInterval));
            rtInfo[ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>::type_info.name] = attribute;
            continue;
        }

        if (is_type<opset1::Convolution>(node)) {
            auto& rtInfo = node->get_input_node_shared_ptr(0)->get_rt_info();
            auto it = rtInfo.find(ngraph::VariantWrapper<QuantizationAlignmentAttributePtr>::type_info.name);
            if (it != rtInfo.end()) {
                auto attributeWrapper = std::dynamic_pointer_cast<ngraph::VariantWrapper<QuantizationAlignmentAttributePtr>>(it->second);
                assert(attributeWrapper != nullptr);
                attributeWrapper->get()->hasToBeAligned = true;
                continue;
            }
        }

        if (!NetworkHelper::isPrecisionPreserved(node)) {
            continue;
        }

        // TODO: limitation: one operation type is used
        std::shared_ptr<ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>> firstExistingIntervalsAttribute;
        std::shared_ptr<ngraph::VariantWrapper<QuantizationAlignmentAttributePtr>> firstExistingValueAttribute;

        //auto getAttribute = [](std::shared_ptr<Node>& inputNode) -> std::shared_ptr<ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>> {
        //    auto& rtInfo = inputNode->get_rt_info();
        //    auto it = rtInfo.find(ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>::type_info.name);
        //    if (it == rtInfo.end()) {
        //        return nullptr;
        //    }

        //    auto tmpAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>>(it->second);
        //    assert(tmpAttribute != nullptr);
        //    return tmpAttribute;
        //};

        // get nodes
        std::vector<std::shared_ptr<ngraph::Node>> inputNodes;
        for (auto index = 0ul; index < node->get_input_size(); ++index) {
            const auto& input = node->input(index);
            auto inputNode = input.get_source_output().get_node_shared_ptr();

            const auto dequantization = NetworkHelper::getDequantization(node, index);
            if (!dequantization.empty() &&
                (is_type<opset1::Convert>(dequantization.data.get_node())) &&
                is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
                inputNode = dequantization.data.get_node()->get_input_node_shared_ptr(0);
            }

            auto existingIntervalsAttribute = getAttribute<IntervalsAlignmentAttributePtr>(inputNode);
            if (existingIntervalsAttribute != nullptr) {
                if (firstExistingIntervalsAttribute == nullptr) {
                    firstExistingIntervalsAttribute = existingIntervalsAttribute;
                }
            }

            auto existingValueAttribute = getAttribute<QuantizationAlignmentAttributePtr>(inputNode);
            if (existingValueAttribute != nullptr) {
                if (firstExistingValueAttribute == nullptr) {
                    firstExistingValueAttribute = existingValueAttribute;
                }
            }

            if (is_type<opset1::FakeQuantize>(inputNode)) {
                auto& rt = node->get_rt_info();
                const std::string& name = ngraph::VariantWrapper<QuantizationAlignmentAttributePtr>::type_info.name;
                if (rt.find(name) == rt.end()) {
                    const auto attribute = std::make_shared<ngraph::VariantWrapper<QuantizationAlignmentAttributePtr>>(
                        std::make_shared<QuantizationAlignmentAttribute>());
                    rt[name] = attribute;
                }
            }

            inputNodes.push_back(inputNode);
        }

        //// merge: share between other operations - implicit backward propagation
        //if (firstExistingIntervalsAttribute != nullptr) {
        //    auto attribute = firstExistingIntervalsAttribute->merge(inputNodes);
        //    auto newAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>>(attribute);
        //    assert(newAttribute != nullptr);

        //    bool wasReplaced = false;
        //    for (size_t i = 1ul; i < inputNodes.size(); i++) {
        //        auto oldAttribute = getAttribute<IntervalsAlignmentAttributePtr>(inputNodes[i]);
        //        if (oldAttribute != nullptr) {
        //            const std::string name = ngraph::VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>>::type_info.name;
        //            replaceAttributeInNodes(f, name, newAttribute, oldAttribute, node);
        //            wasReplaced = true;
        //        }
        //    }
        //    if (!wasReplaced) {
        //        node->get_rt_info()[ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>::type_info.name] = newAttribute;
        //    }
        //}

        mergeAndReplace<IntervalsAlignmentAttributePtr>(f, node, firstExistingIntervalsAttribute, inputNodes);

        mergeAndReplace<QuantizationAlignmentAttributePtr>(f, node, firstExistingValueAttribute, inputNodes);
    }
    return true;
}
