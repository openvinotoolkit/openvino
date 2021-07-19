// Copyright (C) 2021 Intel Corporation
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

template <typename AttributeType>
class ngraph::pass::low_precision::PropagateToInput : public ngraph::pass::MatcherPass {
public:
    PropagateToInput() {
        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto node = m.get_match_root();
            if (transformation_callback(node)) {
                return false;
            }

            {
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "PropagateToInput");

                for (auto input : node->inputs()) {
                    auto parentAttribute = getSourceOutputAttribute(input);
                    if (parentAttribute == nullptr) {
                        continue;
                    }

                    auto attribute = getAttribute<std::shared_ptr<AttributeType>>(input);
                    if (attribute != nullptr) {
                        if ((attribute->get()->sharedValue != nullptr) && (attribute->get()->sharedValue->precisions.empty())) {
                            return false;
                        }

                        std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<AttributeType>>>> attributes = { attribute };
                        parentAttribute->merge(attributes);
                    }

                    auto& rt = input.get_rt_info();
                    rt[ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::type_info.name] = parentAttribute;
                }
            }
            return true;
        };

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(pattern::any_input(), "PropagateThroughPrecisionPreserved");
        this->register_matcher(matcher, callback);
    }

private:
    // TODO: possible duplicate: PropagateThroughPrecisionPreserved::getParentInputRestrictions
    std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>> getSourceOutputAttribute(const Input<Node>& input) {
        auto getInput = [](const Input<Node>& input) {
            const auto dequantization = NetworkHelper::getDequantization(input.get_node()->shared_from_this(), input.get_index());
            if (!dequantization.empty() &&
                is_type<opset1::Convert>(dequantization.data.get_node()) &&
                (dequantization.data.get_node()->get_input_size() == 1ul) &&
                is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
                return dequantization.data.get_node()->input(0);
            }

            return input;
        };

        auto input2 = getInput(input);
        auto output = input2.get_source_output();
        std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>> attribute = getAttributeFromOutput<std::shared_ptr<AttributeType>>(output);
        if (attribute == nullptr) {
            attribute = getAttribute<std::shared_ptr<AttributeType>>(output.get_node_shared_ptr());
        }
        return attribute;
    }

    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>> getParentInputRestrictions(
        const std::shared_ptr<ngraph::Node> node) {
        std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>> parentAttributes;
        for (size_t index = 0ul; index < node->get_input_size(); index++) {
            const Input<Node>& input = node->input(index);
            const auto attribute = getSourceOutputAttribute(input);
            if (attribute != nullptr) {
                parentAttributes.push_back(attribute);
            }
        }
        return parentAttributes;
    }
};
