// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

template <typename AttributeType>
class PropagateAttributeToPrecisionPreserved;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

//using AttributeType = AvgPoolPrecisionPreservedAttribute;

template <typename AttributeType>
class ngraph::pass::low_precision::PropagateAttributeToPrecisionPreserved : public ngraph::pass::MatcherPass {
public:
    PropagateAttributeToPrecisionPreserved() {
        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto node = m.get_match_root();
            if (!node || transformation_callback(node)) {
                return false;
            }

            if (NetworkHelper::isPrecisionPreserved(node)) {
                const auto parentRestrictions = getParentInputRestrictions(node);
                if (parentRestrictions.empty()) {
                    return false;
                }

                auto resultAttribute = parentRestrictions[0];

                std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>> toMerge = parentRestrictions;
                toMerge.erase(toMerge.begin());
                resultAttribute->merge(toMerge);

                for (size_t index = 1ul; index < parentRestrictions.size(); index++) {
                    for (const auto attributeWeakPtr : parentRestrictions[index]->get()->sharedValue->attributes) {
                        auto attribute = attributeWeakPtr.lock();
                        if (attribute == nullptr) {
                            continue;
                        }
                        attribute->sharedValue = resultAttribute->get()->sharedValue;
                        resultAttribute->get()->sharedValue->attributes.push_back(attribute);
                    }
                }

                auto& rt = node->get_rt_info();
                rt[ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::type_info.name] = resultAttribute;
            } else {
                for (auto input : node->inputs()) {
                    auto parentAttribute = getSourceOutputAttribute(input);
                    if (parentAttribute == nullptr) {
                        continue;
                    }

                    auto attribute = getAttribute<std::shared_ptr<AttributeType>>(input);
                    if (attribute != nullptr) {
                        parentAttribute->merge(std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<AttributeType>>>>{ attribute });
                    }

                    auto& rt = input.get_rt_info();
                    rt[ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::type_info.name] = parentAttribute;
                }
            }
            return true;
        };

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(pattern::any_input(), "PropagateAttributeToPrecisionPreserved");
        this->register_matcher(matcher, callback);
    }

    virtual ~PropagateAttributeToPrecisionPreserved() = default;

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

    std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>> getSourceOutputAttribute(const Input<Node>& input) {
        //auto inputNode = input.get_source_output().get_node()->shared_from_this();
        auto input2 = get(input);

        //const auto dequantization = NetworkHelper::getDequantization(input.get_node()->shared_from_this(), input.get_index());
        //if (!dequantization.empty() &&
        //    (is_type<opset1::Convert>(dequantization.data.get_node())) &&
        //    is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
        //    //inputNode = dequantization.data.get_node()->get_input_node_shared_ptr(0);
        //    assert(dequantization.data.get_target_inputs().size() == 1ul);
        //    input2 = *dequantization.data.get_target_inputs().begin();
        //    //input2 = input3;
        //}

        //// ngraph::is_type<AttributeType>(inputNode)
        //// TODO: is_type<AttributeType>(inputNode) - is not tested
        ////getAttribute
        //auto& rt = NetworkHelper::isPrecisionPreserved(inputNode) ?
        //    inputNode->get_rt_info() :
        //    input.get_source_output().get_rt_info();
        //auto it = rt.find(ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::type_info.name);
        //if (it == rt.end()) {
        //    return nullptr;
        //}

        //const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>(it->second);
        //return attribute;

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
            //auto inputNode = input.get_source_output().get_node()->shared_from_this();

            //const auto dequantization = NetworkHelper::getDequantization(node, index);
            //if (!dequantization.empty() &&
            //    (is_type<opset1::Convert>(dequantization.data.get_node())) &&
            //    is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
            //    inputNode = dequantization.data.get_node()->get_input_node_shared_ptr(0);
            //}

            //auto& rt = NetworkHelper::isPrecisionPreserved(inputNode) ? inputNode->get_rt_info() : input.get_source_output().get_rt_info();
            //auto it = rt.find(ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::type_info.name);
            //if (it != rt.end()) {
            //    const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>(it->second);
            //    parentAttributes.push_back(attribute);
            //}

            const auto attribute = getSourceOutputAttribute(input);
            if (attribute != nullptr) {
                parentAttributes.push_back(attribute);
            }
        }
        return parentAttributes;
    }
};
