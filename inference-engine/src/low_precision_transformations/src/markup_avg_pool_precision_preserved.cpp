// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_avg_pool_precision_preserved.hpp"

#include <assert.h>
#include <memory>
#include <unordered_set>
#include <set>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/avg_pool_precision_preserved_attribute.hpp"

using namespace ngraph;

bool ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        // create new
        if (ngraph::is_type<opset1::AvgPool>(node)) {
            auto& rtInfo = node->get_rt_info();

            const auto precisionPreservedAttribute = std::make_shared<ngraph::VariantWrapper<PrecisionPreservedAttributePtr>>(
                make_shared_attribute<PrecisionPreservedAttribute>(false));
            rtInfo[ngraph::VariantWrapper<PrecisionPreservedAttributePtr>::type_info.name] = precisionPreservedAttribute;

            const auto& sharedValue = precisionPreservedAttribute->get()->sharedValue;

            auto v = make_shared_attribute<AvgPoolPrecisionPreservedAttribute>();
            NetworkHelper::reassign<PrecisionPreservedSharedValue, PrecisionPreservedAttribute>(v->sharedValue, sharedValue->attributes);

            const auto attribute = std::make_shared<ngraph::VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>>(v);
            rtInfo[ngraph::VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>::type_info.name] = attribute;

            continue;
        }

        if (!NetworkHelper::isPrecisionPreserved(node)) {
            if (ngraph::is_type<opset1::FakeQuantize>(node)) {
                continue;
            }

            for (const auto& input : node->inputs()) {
                auto inputNode = input.get_source_output().get_node_shared_ptr();
                auto attribute = getAttribute<AvgPoolPrecisionPreservedAttributePtr>(inputNode);
                if (attribute != nullptr) {
                    attribute->get()->sharedValue->value = true;
                }
            }
            continue;
        }

        //// TODO: not implemented: if node already has AvgPoolPrecisionPreservedAttribute with default value then we have to merge them
        //{
        //    auto& rtInfo = node->get_rt_info();
        //    auto it = rtInfo.find(ngraph::VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>::type_info.name);
        //    if (it != rtInfo.end()) {
        //        auto tmpAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>>(it->second);
        //        if (tmpAttribute->get()->sharedValue->value) {
        //            // TODO: not implemented
        //        }
        //    }
        //}

        std::shared_ptr<ngraph::VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>> firstExistingAttribute;

        std::vector<std::shared_ptr<ngraph::Node>> inputNodes;
        for (const auto& input : node->inputs()) {
            auto inputNode = input.get_source_output().get_node_shared_ptr();

            if (firstExistingAttribute == nullptr) {
                auto attribute = getAttribute<AvgPoolPrecisionPreservedAttributePtr>(inputNode);
                if (attribute != nullptr) {
                    firstExistingAttribute = attribute;
                }
            }

            inputNodes.push_back(inputNode);
        }

        if (firstExistingAttribute != nullptr) {
            const bool wasFound = is_type<opset1::FakeQuantize>(node);
            if (wasFound) {
                firstExistingAttribute->get()->sharedValue->value = !firstExistingAttribute->get()->sharedValue->value;
            }

            auto newAttribute = firstExistingAttribute->merge(inputNodes);

            if (!wasFound) {
                auto& rtInfo = node->get_rt_info();
                rtInfo[ngraph::VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>::type_info.name] = newAttribute;
            }
        }

        mergeAndReplace<AvgPoolPrecisionPreservedAttributePtr>(f, node, firstExistingAttribute, inputNodes);
    }
    return true;
}
