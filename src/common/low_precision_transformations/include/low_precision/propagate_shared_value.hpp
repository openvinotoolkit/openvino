// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <memory>
#include <vector>

#include "openvino/core/node.hpp"

#include "low_precision/lpt_visibility.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "low_precision/network_helper.hpp"
#include "lpt_itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

template <class AttributeType>
class LP_TRANSFORMATIONS_API PropagateSharedValue;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief PropagateSharedValue transformation propagates shared value AttributeType attribute instances
 * through precision preserved operations.
 *
 * For more details about the transformation, refer to
 * [PropagateSharedValue](@ref openvino_docs_OV_UG_lpt_PropagateSharedValue) page
 * in the OpenVINO Developer Guide.
 */
template <class AttributeType>
class ov::pass::low_precision::PropagateSharedValue : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("low_precision::PropagateSharedValue");
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "PropagateSharedValue");

        std::vector<std::shared_ptr<ov::Node>> nodes(f->get_ordered_ops());
        for (auto it = nodes.begin(); it != nodes.end(); it++) {
            const std::shared_ptr<Node> node = *it;

            ov::op::util::process_subgraph(*this, node);

            if (ov::is_type<opset1::FakeQuantize>(node)) {
                assert(node->get_output_size() == 1ul);
                auto& outputRtInfo = node->output(0).get_rt_info();
                outputRtInfo[AttributeType::get_type_info_static()] = AttributeType{std::set<element::Type>{element::u8, element::i8}};
                continue;
            }

            if (!NetworkHelper::isPrecisionPreserved(node)) {
                for (auto& input : node->inputs()) {
                    auto parentNode = input.get_source_output().get_node_shared_ptr();

                    auto getAttributes = [](const Input<Node>& nodeInput) {
                        const std::string name = PrecisionsAttribute::get_type_info_static();

                        auto node = nodeInput.get_source_output().get_node_shared_ptr();
                        std::vector<ov::Any> attributes;
                        if (ov::is_type<opset1::FakeQuantize>(node)) {
                            // output
                            auto& rt = nodeInput.get_source_output().get_rt_info();
                            auto it = rt.find(name);
                            if (it != rt.end()) {
                                attributes.push_back(it->second);
                            }
                        }

                        return attributes;
                    };

                    auto& nodeRt = input.get_rt_info();

                    const std::string name = PrecisionsAttribute::get_type_info_static();
                    const auto it = nodeRt.find(name);
                    if (it == nodeRt.end()) {
                        continue;
                    }

                    std::vector<ov::Any> attributes{ it->second };

                    auto parentAttributes = getAttributes(input);
                    if (parentAttributes.empty()) {
                        continue;
                    }

                    for (auto& parentAttribute : parentAttributes) {
                        parentAttribute.as<PrecisionsAttribute>.merge(attributes);
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
    std::vector<ov::Any> getParentInputRestrictions(
        const std::shared_ptr<ov::Node> node) {
        std::vector<ov::Any> parentAttributes;
        for (size_t index = 0ul; index < node->get_input_size(); index++) {
            const Input<Node>& input = node->input(index);
            auto inputNode = input.get_source_output().get_node()->shared_from_this();

            const auto dequantization = NetworkHelper::getDequantization(node, index);
            if (!dequantization.empty() &&
                (ov::is_type<opset1::Convert>(dequantization.data.get_node())) &&
                 ov::is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
                inputNode = dequantization.data.get_node()->get_input_node_shared_ptr(0);
            }

            if (NetworkHelper::isPrecisionPreserved(inputNode)) {
                auto& inputRtInfo = inputNode->get_rt_info();
                auto inputAttributeIt = inputRtInfo.find(PrecisionsAttribute::get_type_info_static());
                if (inputAttributeIt != inputRtInfo.end()) {
                    parentAttributes.push_back(inputAttributeIt->second);
                }
            } else if (ov::is_type<opset1::FakeQuantize>(inputNode)) {
                const auto& outputPortRtInfo = inputNode->outputs()[0].get_rt_info();
                auto attributeIt = outputPortRtInfo.find(PrecisionsAttribute::get_type_info_static());
                if (attributeIt != outputPortRtInfo.end()) {
                    parentAttributes.push_back(attributeIt->second);
                }
            }
        }
        return parentAttributes;
    }

    void handle(std::shared_ptr<ov::Model> f, const std::shared_ptr<ov::Node>& node) {
        const bool precisionPreserved = NetworkHelper::isPrecisionPreserved(node);
        if (precisionPreserved) {
            const auto parentRestrictions = getParentInputRestrictions(node);
            if (parentRestrictions.empty()) {
                return;
            }

            // one operation - one output precision
            // merge parent inputs to one current output
            auto resultAttribute = parentRestrictions[0];

            std::vector<ov::Any> toMerge = parentRestrictions;
            toMerge.erase(toMerge.begin());
            resultAttribute.as<PrecisionsAttribute>().merge(toMerge);

            for (size_t index = 1ul; index < parentRestrictions.size(); index++) {
                NetworkHelper::reassign(
                    resultAttribute.as<PrecisionsAttribute>().attribute->sharedValue,
                    parentRestrictions[index].as<PrecisionsAttribute>().attribute->sharedValue->attributes);
            }

            auto& rt = node->get_rt_info();
            rt[PrecisionsAttribute::get_type_info_static()] = resultAttribute;
        }
    }
};

