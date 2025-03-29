// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/lpt_visibility.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "network_helper.hpp"

namespace ov {
namespace pass {
namespace low_precision {

template <typename AttributeType>
class PropagateToInput;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief PropagateToInput transformation propagates AttributeType shared value attribute instances
 * from parent output ports to consumers input ports.
 *
 * For more details about the transformation, refer to
 * [PropagateToInput](@ref openvino_docs_OV_UG_lpt_PropagateToInput) page
 * in the OpenVINO Developer Guide.
 */
template <typename AttributeType>
class ov::pass::low_precision::PropagateToInput : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("low_precision::PropagateToInput");
    PropagateToInput(const std::vector<ov::element::Type>& defaultPrecisions = {ov::element::u8, ov::element::i8}) {
        ov::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto node = m.get_match_root();
            if (transformation_callback(node)) {
                return false;
            }

            {
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "PropagateToInput");
                // Collect input indexes in groups with the same AttributeType shared value for the corresponding inputs
                std::vector<std::vector<size_t>> groups;
                for (size_t fst_idx = 0; fst_idx < node->get_input_size(); ++fst_idx) {
                    groups.push_back(std::vector<size_t>{fst_idx});
                    const auto attribute_1 = getAttribute<AttributeType>(node->input(fst_idx));
                    if (attribute_1.empty())
                        continue;
                    if ((attribute_1.template as<AttributeType>().attribute->sharedValue != nullptr) &&
                        (attribute_1.template as<AttributeType>().value().empty())) {
                        return false;
                    }

                    size_t count = 0;
                    // Check if next inputs have the same shared value as attribute_1
                    for (size_t sec_idx = fst_idx + 1; sec_idx < node->get_input_size(); ++sec_idx) {
                        const Input<Node>& sec_input = node->input(sec_idx);
                        const auto attribute_2 = getAttribute<AttributeType>(sec_input);
                        if (attribute_2.empty())
                            continue;
                        if ((attribute_2.template as<AttributeType>().attribute->sharedValue != nullptr) &&
                            (attribute_2.template as<AttributeType>().value().empty())) {
                            return false;
                        }

                        if (attribute_1.template as<AttributeType>().attribute->sharedValue ==
                            attribute_2.template as<AttributeType>().attribute->sharedValue) {
                            groups[fst_idx].push_back(sec_idx);
                            ++count;
                        }
                    }
                    fst_idx += count;
                }

                for (const auto& group : groups) {
                    ov::Any res_attr;
                    auto input_attr = getAttribute<AttributeType>(node->input(group[0]));
                    if (!input_attr.empty())
                        res_attr = input_attr;

                    // merge all attributes from inputs and the following source outputs into one in current group
                    for (const auto idx : group) {
                        auto parentAttribute = getSourceOutputAttribute(node->input(idx), defaultPrecisions);
                        if (parentAttribute == nullptr)
                            continue;

                        if (res_attr.empty()) {
                            res_attr = parentAttribute;
                        } else {
                            std::vector<ov::Any> toMerge = {parentAttribute};
                            res_attr.template as<AttributeType>().merge_attributes(toMerge);

                            auto& attributes =
                                parentAttribute.template as<AttributeType>().attribute->sharedValue->getAttributes();
                            for (auto&& attributeWeakPtr : attributes) {
                                auto attribute = attributeWeakPtr.lock();
                                if (attribute == nullptr)
                                    continue;
                                attribute->sharedValue = res_attr.template as<AttributeType>().attribute->sharedValue;
                                res_attr.template as<AttributeType>().attribute->sharedValue->addAttribute(attribute);
                            }
                        }
                    }

                    if (!res_attr.empty())
                        for (const auto idx : group)
                            node->input(idx).get_rt_info()[AttributeType::get_type_info_static()] = res_attr;
                }
            }
            return true;
        };

        auto matcher = std::make_shared<ov::pass::pattern::Matcher>(pattern::any_input(), "PropagateThroughPrecisionPreserved");
        this->register_matcher(matcher, callback);
    }

private:
    // TODO: possible duplicate: PropagateThroughPrecisionPreserved::getParentInputRestrictions
    ov::Any getSourceOutputAttribute(const Input<Node>& input, const std::vector<ov::element::Type>& defaultPrecisions) {
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
        const std::shared_ptr<ov::Node> node) {
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
