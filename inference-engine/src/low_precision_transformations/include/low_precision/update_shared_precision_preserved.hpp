// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/pass/pass.hpp>
#include <ngraph/variant.hpp>

#include "low_precision/network_helper.hpp"
#include "low_precision/lpt_itt.hpp"
#include "low_precision/lpt_visibility.hpp"

namespace ov {
namespace pass {
namespace low_precision {

template <typename AttributeType, typename OperationType>
class UpdateSharedPrecisionPreserved;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

template <typename AttributeType, typename ExpectedAttributeType = AttributeType>
class ov::pass::low_precision::UpdateSharedPrecisionPreserved : public ov::pass::MatcherPass {
public:
    UpdateSharedPrecisionPreserved() {
        ov::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto node = m.get_match_root();

            const bool needToCheckExpectedAttributeType = !std::is_same<ExpectedAttributeType, AttributeType>::value;
            if (!needToCheckExpectedAttributeType) {
                // expected attribute is ignored, set attributes for node inputs except Result & FakeQuantize operations
                if (is_type<ov::opset1::Result>(node) ||
                    is_type<ov::opset1::FakeQuantize>(node) ||
                    transformation_callback(node)) {
                    return false;
                }
            }

            if (ov::pass::low_precision::NetworkHelper::isPrecisionPreserved(node) || is_type<opset1::FakeQuantize>(node)) {
                return false;
            }

            {
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "UpdateSharedPrecisionPreserved");

                // TODO: check if node can be quantized, if not, then doesn't update
                for (auto input : node->inputs()) {
                    auto precisionsAttributeWrapper = getAttribute<PrecisionsAttributePtr>(input);
                    if (precisionsAttributeWrapper != nullptr) {
                        const auto precisionsAttribute = precisionsAttributeWrapper->get();
                        assert(precisionsAttribute != nullptr);
                        if (precisionsAttribute->sharedValue->precisions.empty()) {
                            return false;
                        }
                    }
                }

                for (auto input : node->inputs()) {
                    if (needToCheckExpectedAttributeType) {
                        if (getAttribute<ExpectedAttributeType>(input) == nullptr) {
                            return false;
                        }
                    }
                    auto parentAttribute = getSourceAttribute(input);
                    if (parentAttribute == nullptr) {
                        continue;
                    }

                    parentAttribute->get()->sharedValue->value = true;
                }
            }

            return true;
        };

        auto matcher = std::make_shared<ov::pattern::Matcher>(pattern::any_input(), "PropagateThroughPrecisionPreserved");
        this->register_matcher(matcher, callback);
    }

private:
    Input<Node> getDequantizationInput(const Input<Node>& input) {
        const auto dequantization = NetworkHelper::getDequantization(input.get_node()->shared_from_this(), input.get_index());
        if (!dequantization.empty() &&
            (is_type<opset1::Convert>(dequantization.data.get_node())) &&
            is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
            assert(dequantization.data.get_target_inputs().size() == 1ul);
            return *dequantization.data.get_target_inputs().begin();
        }
        return input;
    }

    std::shared_ptr<ov::VariantWrapper<AttributeType>> getSourceAttribute(const Input<Node>& input) {
        const auto dequantizationInput = getDequantizationInput(input);
        const auto output = dequantizationInput.get_source_output();
        auto attribute = ov::pass::low_precision::getAttribute<AttributeType>(output.get_node()->shared_from_this());
        if (attribute == nullptr) {
            attribute = ov::pass::low_precision::getAttribute<AttributeType>(output.get_node_shared_ptr());
        }
        return attribute;
    }
};
