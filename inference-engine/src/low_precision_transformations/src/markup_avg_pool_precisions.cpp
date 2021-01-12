// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_avg_pool_precisions.hpp"

#include <assert.h>
#include <memory>
#include <unordered_set>
#include <set>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

using namespace ngraph;

bool ngraph::pass::low_precision::MarkupAvgPoolPrecisions::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        // create new
        if (ngraph::is_type<opset1::AvgPool>(node)) {
            auto& rtInfo = node->get_rt_info();
            const auto attribute = std::make_shared<::ngraph::VariantWrapper<PrecisionPreservedAttribute>>(
                PrecisionPreservedAttribute::create<opset1::FakeQuantize>(true));
            rtInfo[ngraph::VariantWrapper<PrecisionPreservedAttribute>::type_info.name] = attribute;
            continue;
        }

        // complete this branch
        //if (!MarkupPrecisions::isPrecisionPreserved(node)) {
        //    continue;
        //}
        {
            auto& rtInfo = node->get_rt_info();
            auto it = rtInfo.find(ngraph::VariantWrapper<PrecisionPreservedAttribute>::type_info.name);
            if (it == rtInfo.end()) {
                continue;
            }
            auto tmpAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionPreservedAttribute>>(it->second);
            if (!tmpAttribute->get().sharedValue->value) {
                continue;
            }
        }

        // TODO: limitation: one operation type is used
        std::shared_ptr<ngraph::VariantWrapper<PrecisionPreservedAttribute>> firstExistingAttribute;

        // get nodes
        std::vector<std::shared_ptr<ngraph::Node>> inputNodes;
        for (const auto& input : node->inputs()) {
            auto inputNode = input.get_source_output().get_node_shared_ptr();

            auto& rtInfo = inputNode->get_rt_info();
            auto it = rtInfo.find(ngraph::VariantWrapper<PrecisionPreservedAttribute>::type_info.name);
            if (it != rtInfo.end()) {
                auto tmpAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionPreservedAttribute>>(it->second);
                assert(tmpAttribute != nullptr);
                if (!tmpAttribute->get().sharedValue->operationName.empty()) {
                    if (firstExistingAttribute == nullptr) {
                        firstExistingAttribute = tmpAttribute;
                    } else {
                        NGRAPH_CHECK(firstExistingAttribute->get().sharedValue->operationName == tmpAttribute->get().sharedValue->operationName, "Only one operation is supported");
                    }
                }
            }

            inputNodes.push_back(inputNode);
        }

        // TODO: not completed: reuse from PropagatePrecision
        // merge: share between other operations - implicit backward propagation
        if (firstExistingAttribute != nullptr) {
            const bool wasFound = firstExistingAttribute->get().sharedValue->operationName == node->get_type_info().name;
            if (wasFound) {
                firstExistingAttribute->get().sharedValue->value = !firstExistingAttribute->get().sharedValue->value;
            }

            auto newAttribute = firstExistingAttribute->merge(inputNodes);

            if (!wasFound) {
                auto& rtInfo = node->get_rt_info();
                rtInfo[ngraph::VariantWrapper<PrecisionPreservedAttribute>::type_info.name] = newAttribute;
            }
        }
    }
    return true;
}
