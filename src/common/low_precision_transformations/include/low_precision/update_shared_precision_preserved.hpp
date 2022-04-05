// Copyright (C) 2018-2022 Intel Corporation
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

namespace ngraph {
namespace pass {
namespace low_precision {

template <typename AttributeType, typename OperationType>
class UpdateSharedPrecisionPreserved;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief UpdateSharedPrecisionPreserved transformation updates shared AttributeType attribute instance value to true
 * for precision preserved operations if ExpectedAttributeType exist.
 *
 * For more details about the transformation, refer to
 * [UpdateSharedPrecisionPreserved](@ref openvino_docs_OV_UG_lpt_UpdateSharedPrecisionPreserved) page
 * in the Inference Engine Developer Guide.
 */
template <typename AttributeType, typename ExpectedAttributeType = AttributeType>
class ngraph::pass::low_precision::UpdateSharedPrecisionPreserved : public ngraph::pass::MatcherPass {
public:
    UpdateSharedPrecisionPreserved(const std::vector<ngraph::element::Type>& defaultPrecisions = precision_set::int8_support) {
        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto node = m.get_match_root();

            const bool needToCheckExpectedAttributeType = !std::is_same<ExpectedAttributeType, AttributeType>::value;
            if (!needToCheckExpectedAttributeType) {
                // expected attribute is ignored, set attributes for node inputs except Result & FakeQuantize operations
                if (ov::is_type<ngraph::opset1::Result>(node) ||
                    ov::is_type<ngraph::opset1::FakeQuantize>(node) ||
                    transformation_callback(node)) {
                    return false;
                }
            }

            if (ngraph::pass::low_precision::NetworkHelper::isPrecisionPreserved(node) || ov::is_type<opset1::FakeQuantize>(node)) {
                return false;
            }

            {
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "UpdateSharedPrecisionPreserved");

                // TODO: check if node can be quantized, if not, then doesn't update
                for (auto input : node->inputs()) {
                    auto precisionsAttributeWrapper = getAttribute<PrecisionsAttribute>(input);
                    if (!precisionsAttributeWrapper.empty()) {
                        if (precisionsAttributeWrapper.as<PrecisionsAttribute>().value().empty()) {
                            return false;
                        }
                    }
                }

                for (auto input : node->inputs()) {
                    if (needToCheckExpectedAttributeType) {
                        const auto& attribute = getAttribute<ExpectedAttributeType>(input);
                        if (attribute.empty()) {
                            return false;
                        }

                        const auto& expectedAttribute = attribute.template as<ExpectedAttributeType>();
                        if (expectedAttribute.is_skipped()) {
                            return false;
                        }
                    }
                    auto parentAttribute = getSourceAttribute(input, defaultPrecisions);
                    if (parentAttribute.empty()) {
                        continue;
                    }
                    parentAttribute.template as<AttributeType>().value() = true;
                }
            }

            return true;
        };

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(pattern::any_input(), "UpdateSharedPrecisionPreserved");
        this->register_matcher(matcher, callback);
    }

private:
    Input<Node> getDequantizationInput(const Input<Node>& input, const std::vector<ngraph::element::Type>& defaultPrecisions) {
        const auto dequantization = NetworkHelper::getDequantization(input.get_node()->shared_from_this(), defaultPrecisions, input.get_index());
        if (!dequantization.empty() &&
            (ov::is_type<opset1::Convert>(dequantization.data.get_node())) &&
            ov::is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
            assert(dequantization.data.get_target_inputs().size() == 1ul);
            return *dequantization.data.get_target_inputs().begin();
        }
        return input;
    }

    ov::Any getSourceAttribute(const Input<Node>& input, const std::vector<ngraph::element::Type>& defaultPrecisions) {
        const auto dequantizationInput = getDequantizationInput(input, defaultPrecisions);
        const auto output = dequantizationInput.get_source_output();
        auto attribute = ngraph::pass::low_precision::getAttribute<AttributeType>(output.get_node()->shared_from_this());
        if (attribute.empty()) {
            attribute = ngraph::pass::low_precision::getAttribute<AttributeType>(output.get_node_shared_ptr());
        }
        return attribute;
    }
};
