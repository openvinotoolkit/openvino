// Copyright (C) 2018-2022 Intel Corporation
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

/**
 * @ingroup ie_transformation_common_api
 * @brief PropagateThroughPrecisionPreserved transformation propagates AttributeType attribute instances
 * through precision preserved operations.
 *
 * For more details about the transformation, refer to
 * [PropagateThroughPrecisionPreserved](@ref openvino_docs_OV_UG_lpt_PropagateThroughPrecisionPreserved) page
 * in the Inference Engine Developer Guide.
 */
template <typename AttributeType>
class ngraph::pass::low_precision::PropagateThroughPrecisionPreserved : public ngraph::pass::MatcherPass {
public:
    PropagateThroughPrecisionPreserved(const std::vector<ngraph::element::Type>& defaultPrecisions = precision_set::int8_support) {
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

                const auto parentRestrictions = getParentInputRestrictions(node, defaultPrecisions);
                if (parentRestrictions.empty()) {
                    return false;
                }

                auto& resultAttribute = parentRestrictions[0].template as<AttributeType>();

                std::vector<ov::Any> toMerge = parentRestrictions;
                // TODO: LPT: handle pointer on itself in IntervalsAlignmentAttribute::merge and remove erase, task #59498
                toMerge.erase(toMerge.begin());
                const_cast<AttributeType&>(resultAttribute).merge(toMerge);

                for (size_t index = 1ul; index < parentRestrictions.size(); index++) {
                    auto& attributes = parentRestrictions[index].template as<AttributeType>().attribute->sharedValue->getAttributes();
                    for (auto&& attributeWeakPtr : attributes) {
                        auto attribute = attributeWeakPtr.lock();
                        if (attribute == nullptr) {
                            continue;
                        }
                        attribute->sharedValue = resultAttribute.attribute->sharedValue;
                        resultAttribute.attribute->sharedValue->addAttribute(attribute);
                    }
                }

                auto &rt = node->get_rt_info();
                rt[AttributeType::get_type_info_static()] = resultAttribute;
            }
            return true;
        };

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(pattern::any_input(), "PropagateThroughPrecisionPreserved");
        this->register_matcher(matcher, callback);
    }

private:
    ov::Any getSourceOutputAttribute(const Input<Node>& input) {
        auto input2 = input;
        auto output = input2.get_source_output();
        auto attribute = getAttributeFromOutput<AttributeType>(output);
        if (attribute.empty()) {
            attribute = getAttribute<AttributeType>(output.get_node_shared_ptr());
        }
        return attribute;
    }

    // TODO: possible duplicate: PropagateToInput::getSourceOutputAttribute
    std::vector<ov::Any> getParentInputRestrictions(const std::shared_ptr<ngraph::Node> node,
        const std::vector<ngraph::element::Type>& defaultPrecisions) {
        std::vector<ov::Any> parentAttributes;
        auto getInput = [&defaultPrecisions](const std::shared_ptr<ngraph::Node>& node, const size_t index) -> Input<Node> {
            const auto dequantization = NetworkHelper::getDequantization(node, defaultPrecisions, index);
            if (!dequantization.empty() &&
                ov::is_type<opset1::Convert>(dequantization.data.get_node()) &&
                (dequantization.data.get_node()->get_input_size() == 1ul) &&
                ov::is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
                return dequantization.data.get_node()->input(0);
            }

            return node->input(index);
        };

        for (size_t index = 0ul; index < node->get_input_size(); index++) {
            const Input<Node>& input = getInput(node, index);
            const auto attribute = getSourceOutputAttribute(input);
            if (!attribute.empty()) {
                parentAttributes.push_back(attribute);
            }
        }

        return parentAttributes;
    }
};
