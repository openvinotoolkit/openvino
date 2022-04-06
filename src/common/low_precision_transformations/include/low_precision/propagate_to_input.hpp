// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

template <typename AttributeType>
class PropagateToInput;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief PropagateToInput transformation propagates AttributeType shared value attribute instances
 * from parent output ports to consumers input ports.
 *
 * For more details about the transformation, refer to
 * [PropagateToInput](@ref openvino_docs_OV_UG_lpt_PropagateToInput) page
 * in the Inference Engine Developer Guide.
 */
template <typename AttributeType>
class ngraph::pass::low_precision::PropagateToInput : public ngraph::pass::MatcherPass {
public:
    PropagateToInput(const std::vector<ngraph::element::Type>& defaultPrecisions = { ngraph::element::u8, ngraph::element::i8 }) {
        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto node = m.get_match_root();
            if (transformation_callback(node)) {
                return false;
            }

            {
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "PropagateToInput");

                for (auto input : node->inputs()) {
                    auto parentAttribute = getSourceOutputAttribute(input, defaultPrecisions);
                    if (parentAttribute == nullptr) {
                        continue;
                    }

                    auto attribute = getAttribute<AttributeType>(input);
                    if (!attribute.empty()) {
                        if ((attribute.template as<AttributeType>().attribute->sharedValue != nullptr) &&
                            (attribute.template as<AttributeType>().value().empty())) {
                            return false;
                        }

                        std::vector<ov::Any> attributes = { attribute };
                        parentAttribute.template as<AttributeType>().merge(attributes);
                    }

                    auto& rt = input.get_rt_info();
                    rt[AttributeType::get_type_info_static()] = parentAttribute;
                }
            }
            return true;
        };

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(pattern::any_input(), "PropagateThroughPrecisionPreserved");
        this->register_matcher(matcher, callback);
    }

private:
    // TODO: possible duplicate: PropagateThroughPrecisionPreserved::getParentInputRestrictions
    ov::Any getSourceOutputAttribute(const Input<Node>& input, const std::vector<ngraph::element::Type>& defaultPrecisions) {
        auto getInput = [&defaultPrecisions](const Input<Node>& input) {
            const auto dequantization = NetworkHelper::getDequantization(input.get_node()->shared_from_this(), defaultPrecisions, input.get_index());
            if (!dequantization.empty() &&
                ov::is_type<opset1::Convert>(dequantization.data.get_node()) &&
                (dequantization.data.get_node()->get_input_size() == 1ul) &&
                ov::is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
                return dequantization.data.get_node()->input(0);
            }

            return input;
        };

        auto input2 = getInput(input);
        auto output = input2.get_source_output();
        auto attribute = getAttributeFromOutput<AttributeType>(output);
        if (attribute.empty()) {
            attribute = getAttribute<AttributeType>(output.get_node_shared_ptr());
        }
        return attribute;
    }

    std::vector<ov::Any> getParentInputRestrictions(
        const std::shared_ptr<ngraph::Node> node) {
        std::vector<ov::Any> parentAttributes;
        for (size_t index = 0ul; index < node->get_input_size(); index++) {
            const Input<Node>& input = node->input(index);
            const auto attribute = getSourceOutputAttribute(input);
            if (!attribute.empty()) {
                parentAttributes.push_back(attribute);
            }
        }
        return parentAttributes;
    }
};
