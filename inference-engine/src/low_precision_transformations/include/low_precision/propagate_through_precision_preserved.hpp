// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/variant.hpp>

#include "low_precision/lpt_visibility.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/lpt_itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

template <typename AttributeType>
class PropagateThroughPrecisionPreserved;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

template <typename AttributeType>
class ngraph::pass::low_precision::PropagateThroughPrecisionPreserved : public ngraph::pass::MatcherPass {
public:
    PropagateThroughPrecisionPreserved() {
        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto node = m.get_match_root();
            if (transformation_callback(node)) {
                return false;
            }

            {
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "PropagateThroughPrecisionPreserved");

                if (!ngraph::pass::low_precision::NetworkHelper::isPrecisionPreserved(node)) {
                    return false;
                }

                const auto parentRestrictions = getParentInputRestrictions(node);
                if (parentRestrictions.empty()) {
                    return false;
                }

                auto resultAttribute = parentRestrictions[0];

                std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>> toMerge = parentRestrictions;
                // TODO: LPT: handle pointer on itself in VariantWrapper<IntervalsAlignmentAttributePtr>::merge and remove erase, task #59498
                toMerge.erase(toMerge.begin());
                resultAttribute->merge(toMerge);

                for (size_t index = 1ul; index < parentRestrictions.size(); index++) {
                    const auto attributes = parentRestrictions[index]->get()->sharedValue->attributes;
                    for (const auto attributeWeakPtr : attributes) {
                        auto attribute = attributeWeakPtr.lock();
                        if (attribute == nullptr) {
                            continue;
                        }
                        attribute->sharedValue = resultAttribute->get()->sharedValue;
                        resultAttribute->get()->sharedValue->attributes.push_back(attribute);
                    }
                }

                auto &rt = node->get_rt_info();
                rt[ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::type_info.name] = resultAttribute;
            }
            return true;
        };

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(pattern::any_input(), "PropagateThroughPrecisionPreserved");
        this->register_matcher(matcher, callback);
    }

private:
    std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>> getSourceOutputAttribute(const Input<Node>& input) {
        auto input2 = input;
        auto output = input2.get_source_output();
        std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>> attribute = getAttributeFromOutput<std::shared_ptr<AttributeType>>(output);
        if (attribute == nullptr) {
            attribute = getAttribute<std::shared_ptr<AttributeType>>(output.get_node_shared_ptr());
        }
        return attribute;
    }

    // TODO: possible duplicate: PropagateToInput::getSourceOutputAttribute
    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>> getParentInputRestrictions(
        const std::shared_ptr<ngraph::Node> node) {
        std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>> parentAttributes;
        auto getInput = [](const std::shared_ptr<ngraph::Node>& node, const size_t index) -> Input<Node> {
            const auto dequantization = NetworkHelper::getDequantization(node, index);
            if (!dequantization.empty() &&
                is_type<opset1::Convert>(dequantization.data.get_node()) &&
                (dequantization.data.get_node()->get_input_size() == 1ul) &&
                is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
                return dequantization.data.get_node()->input(0);
            }

            return node->input(index);
        };

        for (size_t index = 0ul; index < node->get_input_size(); index++) {
            const Input<Node>& input = getInput(node, index);
            const auto attribute = getSourceOutputAttribute(input);
            if (attribute != nullptr) {
                parentAttributes.push_back(attribute);
            }
        }

        return parentAttributes;
    }
};
