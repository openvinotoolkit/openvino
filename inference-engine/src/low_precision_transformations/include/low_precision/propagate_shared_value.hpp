// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "low_precision/network_helper.hpp"
#include "lpt_itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

template <class AttributeType>
class LP_TRANSFORMATIONS_API PropagateSharedValue;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

template <class AttributeType>
class ngraph::pass::low_precision::PropagateSharedValue : public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "PropagateSharedValue");

        std::vector<std::shared_ptr<ngraph::Node>> nodes(f->get_ordered_ops());
        for (auto it = nodes.begin(); it != nodes.end(); it++) {
            const std::shared_ptr<Node> node = *it;
            if (is_type<opset1::FakeQuantize>(node)) {
                assert(node->get_output_size() == 1ul);
                auto& outputRtInfo = node->output(0).get_rt_info();

                auto attribute = make_shared_attribute<AttributeType>(std::set<element::Type>{element::u8, element::i8});

                auto attributeWrapper = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attribute);
                outputRtInfo[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = attributeWrapper;
                continue;
            }

            if (!NetworkHelper::isPrecisionPreserved(node)) {
                for (auto& input : node->inputs()) {
                    auto parentNode = input.get_source_output().get_node_shared_ptr();

                    auto getAttributes = [](const Input<Node>& nodeInput) {
                        const std::string name = ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name;

                        auto node = nodeInput.get_source_output().get_node_shared_ptr();
                        std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> attributes;
                        if (is_type<opset1::FakeQuantize>(node)) {
                            // output
                            auto& rt = nodeInput.get_source_output().get_rt_info();
                            auto it = rt.find(name);
                            if (it != rt.end()) {
                                const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
                                attributes.push_back(attribute);
                            }
                        }

                        return attributes;
                    };

                    auto& nodeRt = input.get_rt_info();

                    const std::string name = ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name;
                    const auto it = nodeRt.find(name);
                    if (it == nodeRt.end()) {
                        continue;
                    }

                    const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
                    std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> attributes{ attribute };

                    auto parentAttributes = getAttributes(input);
                    if (parentAttributes.empty()) {
                        continue;
                    }

                    for (auto& parentAttribute : parentAttributes) {
                        parentAttribute->merge(attributes);
                    }

                    nodeRt[name] = parentAttributes[0];
                }
                continue;
            }

            handle(f, node);
        }
        return true;
    }

private:
    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> getParentInputRestrictions(
        const std::shared_ptr<ngraph::Node> node) {
        std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> parentAttributes;
        for (size_t index = 0ul; index < node->get_input_size(); index++) {
            const Input<Node>& input = node->input(index);
            auto inputNode = input.get_source_output().get_node()->shared_from_this();

            const auto dequantization = NetworkHelper::getDequantization(node, index);
            if (!dequantization.empty() &&
                (is_type<opset1::Convert>(dequantization.data.get_node())) &&
                is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
                inputNode = dequantization.data.get_node()->get_input_node_shared_ptr(0);
            }

            if (NetworkHelper::isPrecisionPreserved(inputNode)) {
                auto& inputRtInfo = inputNode->get_rt_info();
                auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
                if (inputAttributeIt != inputRtInfo.end()) {
                    const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(inputAttributeIt->second);
                    parentAttributes.push_back(attribute);
                }
            } else if (is_type<opset1::FakeQuantize>(inputNode)) {
                const auto& outputPortRtInfo = inputNode->outputs()[0].get_rt_info();
                auto attributeIt = outputPortRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
                if (attributeIt != outputPortRtInfo.end()) {
                    const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attributeIt->second);
                    parentAttributes.push_back(attribute);
                }
            }
        }
        return parentAttributes;
    }

    void handle(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::Node>& node) {
        const bool precisionPreserved = NetworkHelper::isPrecisionPreserved(node);
        if (precisionPreserved) {
            const auto parentRestrictions = getParentInputRestrictions(node);
            if (parentRestrictions.empty()) {
                return;
            }

            // one operation - one output precision
            // merge parent inputs to one current output
            auto resultAttribute = parentRestrictions[0];

            std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> toMerge = parentRestrictions;
            toMerge.erase(toMerge.begin());
            resultAttribute->merge(toMerge);

            for (size_t index = 1ul; index < parentRestrictions.size(); index++) {
                const auto oldAttribute = parentRestrictions[index]->get();
                NetworkHelper::reassign<PrecisionsSharedValue, PrecisionsAttribute>(
                    resultAttribute->get()->sharedValue,
                    parentRestrictions[index]->get()->sharedValue->attributes);
            }

            auto& rt = node->get_rt_info();
            rt[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = resultAttribute;
        }
    }
};

