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
#include "lpt_itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

template <typename AttributeType, typename OperationType>
class UpdateSharedPrecisionPreserved;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

template <typename AttributeType, typename ExpectedAttributeType = AttributeType>
class ngraph::pass::low_precision::UpdateSharedPrecisionPreserved : public ngraph::pass::MatcherPass {
public:
    UpdateSharedPrecisionPreserved() {
        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto node = m.get_match_root();

            const bool needToCheckExpectedAttributeType = !std::is_same<ExpectedAttributeType, AttributeType>::value;
            if (!needToCheckExpectedAttributeType) {
                // expected attribute is ignored, set attributes for node inputs except Result & FakeQuantize operations
                if (is_type<ngraph::opset1::Result>(node) ||
                    is_type<ngraph::opset1::FakeQuantize>(node) ||
                    transformation_callback(node)) {
                    return false;
                }
            }

            if (ngraph::pass::low_precision::NetworkHelper::isPrecisionPreserved(node) || is_type<opset1::FakeQuantize>(node)) {
                return false;
            }

            {
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "UpdateSharedPrecisionPreserved");
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

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(pattern::any_input(), "PropagateThroughPrecisionPreserved");
        this->register_matcher(matcher, callback);
    }

private:
    Input<Node> get(const Input<Node>& input) {
        const auto dequantization = NetworkHelper::getDequantization(input.get_node()->shared_from_this(), input.get_index());
        if (!dequantization.empty() &&
            (is_type<opset1::Convert>(dequantization.data.get_node())) &&
            is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
            //inputNode = dequantization.data.get_node()->get_input_node_shared_ptr(0);
            assert(dequantization.data.get_target_inputs().size() == 1ul);
            return *dequantization.data.get_target_inputs().begin();
        }

        return input;
    }

    std::shared_ptr<ngraph::VariantWrapper<AttributeType>> getSourceAttribute(const Input<Node>& input) {
        // TODO: do we really need it?
        auto input2 = get(input);

        auto output = input2.get_source_output();
        auto attribute = ngraph::pass::low_precision::getAttribute<AttributeType>(output.get_node()->shared_from_this());
        if (attribute == nullptr) {
            // TODO: do we really need it?
            attribute = getAttribute<AttributeType>(output.get_node_shared_ptr());
        }
        return attribute;
    }

//    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>> getParentInputRestrictions(
//        const std::shared_ptr<ngraph::Node> node) {
//        std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>> parentAttributes;
//        for (size_t index = 0ul; index < node->get_input_size(); index++) {
//            const Input<Node>& input = node->input(index);
//            //auto inputNode = input.get_source_output().get_node()->shared_from_this();
//
//            //const auto dequantization = NetworkHelper::getDequantization(node, index);
//            //if (!dequantization.empty() &&
//            //    (is_type<opset1::Convert>(dequantization.data.get_node())) &&
//            //    is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
//            //    inputNode = dequantization.data.get_node()->get_input_node_shared_ptr(0);
//            //}
//
//            //auto& rt = NetworkHelper::isPrecisionPreserved(inputNode) ? inputNode->get_rt_info() : input.get_source_output().get_rt_info();
//            //auto it = rt.find(ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::type_info.name);
//            //if (it != rt.end()) {
//            //    const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>(it->second);
//            //    parentAttributes.push_back(attribute);
//            //}
//
//            const auto attribute = getSourceOutputAttribute(input);
//            if (attribute != nullptr) {
//                parentAttributes.push_back(attribute);
//            }
//        }
//        return parentAttributes;
//    }
};
