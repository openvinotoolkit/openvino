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
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

using namespace ngraph;

bool ngraph::pass::low_precision::AlignConcatQuantizationParamters::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        // create new
        if (ngraph::is_type<opset1::FakeQuantize>(node)) {
            // TODO: FakeQuantize validation is skipped: if FakeQuantize will be not handled then ignore it
            const std::vector<float> lowIntervals = as_type<opset1::Constant>(node->get_input_node_ptr(3))->cast_vector<float>();
            const float lowInterval = *std::min_element(lowIntervals.begin(), lowIntervals.end());

            const auto& highIntervals = as_type<opset1::Constant>(node->get_input_node_ptr(4))->cast_vector<float>();
            const float highInterval = *std::max_element(highIntervals.begin(), highIntervals.end());

            auto& rtInfo = node->get_rt_info();

            const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(ngraph::as_type_ptr<opset1::FakeQuantize>(node));
            const LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(quantizationDetails);

            const auto attribute = std::make_shared<::ngraph::VariantWrapper<QuantizationAlignmentAttribute>>(
                QuantizationAlignmentAttribute(lowInterval, highInterval/*, precisionDetailsAtOutputIntervals.precision*/));
            rtInfo[ngraph::VariantWrapper<QuantizationAlignmentAttribute>::type_info.name] = attribute;
            continue;
        }

        if (is_type<opset1::Convolution>(node)) {
            auto& rtInfo = node->get_input_node_shared_ptr(0)->get_rt_info();
            auto it = rtInfo.find(ngraph::VariantWrapper<QuantizationAlignmentAttribute>::type_info.name);
            if (it != rtInfo.end()) {
                auto attributeWrapper = std::dynamic_pointer_cast<ngraph::VariantWrapper<QuantizationAlignmentAttribute>>(it->second);
                assert(attributeWrapper != nullptr);
                attributeWrapper->get().sharedPart->value->hasToBeAligned = true;
                continue;
            }
        }

        if (!NetworkHelper::isPrecisionPreserved(node)) {
            continue;
        }

        // TODO: limitation: one operation type is used
        std::shared_ptr<ngraph::VariantWrapper<QuantizationAlignmentAttribute>> firstExistingAttribute;

        // get nodes
        std::vector<std::shared_ptr<ngraph::Node>> inputNodes;
        for (const auto& input : node->inputs()) {
            auto inputNode = input.get_source_output().get_node_shared_ptr();

            auto& rtInfo = inputNode->get_rt_info();
            auto it = rtInfo.find(ngraph::VariantWrapper<QuantizationAlignmentAttribute>::type_info.name);
            if (it != rtInfo.end()) {
                auto tmpAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<QuantizationAlignmentAttribute>>(it->second);
                assert(tmpAttribute != nullptr);

                if (firstExistingAttribute == nullptr) {
                    firstExistingAttribute = tmpAttribute;
                }
            }

            inputNodes.push_back(inputNode);
        }

        // merge: share between other operations - implicit backward propagation
        if (firstExistingAttribute != nullptr) {
            auto newAttribute = firstExistingAttribute->merge(inputNodes);
            auto& rtInfo = node->get_rt_info();
            rtInfo[ngraph::VariantWrapper<QuantizationAlignmentAttribute>::type_info.name] = newAttribute;
        }
    }
    return true;
}
