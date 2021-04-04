// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/propagate_precisions.hpp"

#include <assert.h>
#include <deque>
#include <memory>
#include <unordered_map>
#include <set>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

using namespace ngraph;

// 0, 1 - backward propagation, restriction operations begin
// 2 - forward propagation, FakeQuantize operations begin
#define TYPE 2

bool isPrecisionPreserved(std::shared_ptr<Node> node) {
    auto& rtInfo = node->get_rt_info();
    auto it = rtInfo.find(ngraph::VariantWrapper<PrecisionPreservedAttribute>::type_info.name);
    if (it == rtInfo.end()) {
        return false;
    }

    auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionPreservedAttribute>>(it->second);
    return attribute->get().sharedValue->value;
}

#if TYPE == 0

void handle(const std::shared_ptr<ngraph::Node>& node) {
    bool outputRestrictionsInitialized = false;
    std::set<ngraph::element::Type> outputRestrictions;

    for (Output<Node>& output : node->outputs()) {
        //bool outputRestrictionsInitialized = false;

        const auto& inputs = output.get_target_inputs();
        for (const Input<Node>& input : inputs) {
            auto& inputRtInfo = input.get_rt_info();
            auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
            if (inputAttributeIt != inputRtInfo.end()) {
                std::shared_ptr<ngraph::VariantWrapper<PrecisionsAttribute>> inputPrecisionsAttribute =
                    std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(inputAttributeIt->second);

                // TODO: variant #1: merge attributes: not abvious how interpret `nodes`
                // virtual std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector & nodes);
                //inputPrecisionsAttribute->merge();

                // TODO: variant #2: merge manually
                // TODO: need tests
                std::set<ngraph::element::Type> inputPrecisions = inputPrecisionsAttribute->get();
                if (inputPrecisions.empty()) {
                    continue;
                }
                if (outputRestrictionsInitialized) {
                    auto it = outputRestrictions.begin();
                    while (it != outputRestrictions.end()) {
                        if (inputPrecisions.find(*it) == inputPrecisions.end()) {
                            auto itNext = it;
                            itNext++;
                            outputRestrictions.erase(it);
                            it = itNext;
                        } else {
                            it++;
                        }
                    }

                } else {
                    outputRestrictionsInitialized = true;
                    outputRestrictions.insert(inputPrecisions.begin(), inputPrecisions.end());
                }
            }
        }

        //if (outputRestrictionsInitialized) {
        //    auto& outputRtInfo = output.get_rt_info();
        //    outputRtInfo.emplace(
        //        ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name,
        //        std::make_shared<::ngraph::VariantWrapper<PrecisionsAttribute>>(outputRestrictions));
        //}
    }

    if (outputRestrictionsInitialized) {
        if (is_type<opset1::FakeQuantize>(node)) {
            auto& outputs = node->outputs();
            Output<Node>& output = outputs[0];
            auto& outputRtInfo = output.get_rt_info();
            outputRtInfo.emplace(
                ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name,
                std::make_shared<::ngraph::VariantWrapper<PrecisionsAttribute>>(outputRestrictions));
        } else {
            for (Input<Node>& input : node->inputs()) {
                auto& outputRtInfo = input.get_rt_info();
                outputRtInfo.emplace(
                    ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name,
                    std::make_shared<::ngraph::VariantWrapper<PrecisionsAttribute>>(outputRestrictions));
            }
        }
    }
}

bool ngraph::pass::low_precision::PropagatePrecisions::run_on_function(std::shared_ptr<ngraph::Function> f) {
    std::deque<std::shared_ptr<Node>> nodes;
    std::set<std::shared_ptr<Node>> visited;
    for (auto& r : f->get_results()) {
        nodes.push_back(r);
    }

    for (auto& r : f->get_sinks()) {
        nodes.emplace_back(r);
    }

    while (!nodes.empty()) {
        auto curr_node = nodes.front();
        nodes.pop_front();

        if (visited.count(curr_node) || is_type<op::Constant>(curr_node)) {
            continue;
        }

        visited.insert(curr_node);

        std::cout << "PropagatePrecisions::run_on_function: " << curr_node->get_type_name() << ": " << curr_node->get_friendly_name() << std::endl;
        if (is_type<opset1::FakeQuantize>(curr_node) || isPrecisionPreserved(curr_node)) {
            handle(curr_node);
        }

        for (auto& input_value : curr_node->input_values()) {
            const auto& input_node = input_value.get_node_shared_ptr();
            nodes.push_front(input_node);
        }
    }
    return true;
}

#elif TYPE == 1

void handle(const std::shared_ptr<ngraph::Node>& node) {
    std::vector<std::shared_ptr<ngraph::VariantWrapper<PrecisionsAttribute>>> childrenAttributes;
    for (Output<Node>& output : node->outputs()) {
        const auto& inputs = output.get_target_inputs();
        for (const Input<Node>& input : inputs) {
            auto& inputRtInfo = input.get_rt_info();
            auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
            if (inputAttributeIt != inputRtInfo.end()) {
                std::shared_ptr<ngraph::VariantWrapper<PrecisionsAttribute>> inputAttribute =
                    std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(inputAttributeIt->second);

                childrenAttributes.push_back(inputAttribute);
            }
        }
    }

    // TODO: manual merge
    if (!childrenAttributes.empty()) {
        const auto resultAttribute = std::make_shared<ngraph::VariantWrapper<PrecisionsAttribute>>(childrenAttributes[0]->get().sharedPart->value);
        for (size_t index = 1ul; index < childrenAttributes.size(); index++) {
            childrenAttributes[index]->get().sharedPart->value = resultAttribute->get().sharedPart->value;
        }

        if (is_type<opset1::FakeQuantize>(node)) {
            // target operation: update output ports
            auto& outputs = node->outputs();
            assert(outputs.size() == 1ul);
            Output<Node>& output = outputs[0];
            auto& outputRtInfo = output.get_rt_info();
            outputRtInfo.emplace(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name, resultAttribute);
        } else {
            // propagation: define inputs
            for (Input<Node>& input : node->inputs()) {
                auto& outputRtInfo = input.get_rt_info();
                outputRtInfo.emplace(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name, resultAttribute);
            }
        }
    }
}

bool ngraph::pass::low_precision::PropagatePrecisions::run_on_function(std::shared_ptr<ngraph::Function> f) {
    std::vector<std::shared_ptr<ngraph::Node>> nodes(f->get_ordered_ops());
    for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
        const std::shared_ptr<Node> node = *it;
        std::cout << "PropagatePrecisions::run_on_function: " << node->get_type_name() << ": " << node->get_friendly_name() << std::endl;
        if (is_type<opset1::FakeQuantize>(node) || isPrecisionPreserved(node)) {
            handle(node);
        }
    }
    return true;
}

#elif TYPE == 2

std::vector<std::shared_ptr<ngraph::VariantWrapper<PrecisionsAttribute>>> getParentInputRestrictions(const std::shared_ptr<ngraph::Node> node) {
    std::vector<std::shared_ptr<ngraph::VariantWrapper<PrecisionsAttribute>>> parentAttributes;
    for (Input<Node>& input : node->inputs()) {
        const auto& inputNode = input.get_source_output().get_node()->shared_from_this();
        if (isPrecisionPreserved(inputNode)) {
            for (const Input<Node>& input : inputNode->inputs()) {
                auto& inputRtInfo = input.get_rt_info();
                auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
                if (inputAttributeIt != inputRtInfo.end()) {
                    const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(inputAttributeIt->second);
                    parentAttributes.push_back(attribute);
                }
            }
        } else if (is_type<opset1::FakeQuantize>(inputNode)) {
            const auto& outputPortRtInfo = inputNode->outputs()[0].get_rt_info();
            auto attributeIt = outputPortRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
            if (attributeIt != outputPortRtInfo.end()) {
                const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(attributeIt->second);
                parentAttributes.push_back(attribute);
            }
        }
    }
    return parentAttributes;
}

std::shared_ptr<ngraph::VariantWrapper<PrecisionsAttribute>> getParentInputRestriction(const std::shared_ptr<ngraph::Node> node, const size_t parentIndex) {
    const auto& inputs = node->inputs();
    Input<Node> input = inputs[parentIndex];
    const auto& inputNode = input.get_source_output().get_node()->shared_from_this();
    if (isPrecisionPreserved(inputNode)) {
        for (const Input<Node>& input : inputNode->inputs()) {
            auto& inputRtInfo = input.get_rt_info();
            auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
            if (inputAttributeIt != inputRtInfo.end()) {
                const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(inputAttributeIt->second);
                return attribute;
            }
        }
    }

    if (is_type<opset1::FakeQuantize>(inputNode)) {
        const auto& outputPortRtInfo = inputNode->outputs()[0].get_rt_info();
        auto attributeIt = outputPortRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
        if (attributeIt != outputPortRtInfo.end()) {
            const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(attributeIt->second);
            return attribute;
        }
    }

    return nullptr;
}


std::vector<std::shared_ptr<ngraph::VariantWrapper<PrecisionsAttribute>>> getParentOutputRestrictions(const std::shared_ptr<ngraph::Node> node) {
    std::vector<std::shared_ptr<ngraph::VariantWrapper<PrecisionsAttribute>>> parentOutputAttributes;
    for (Input<Node>& input : node->inputs()) {
        const auto& inputNode = input.get_source_output().get_node()->shared_from_this();
        const auto& parentOutput = input.get_source_output();

        auto& parentOutputRtInfo = parentOutput.get_rt_info();
        auto outputAttributeIt = parentOutputRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
        if (outputAttributeIt != parentOutputRtInfo.end()) {
            const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(outputAttributeIt->second);
            parentOutputAttributes.push_back(attribute);
        }
    }
    return parentOutputAttributes;
}

void handle(const std::shared_ptr<ngraph::Node>& node) {
    // TODO: possible need to add validation here to avoid not neccaassary actions for not preserved operations without precision limitations
    const bool precisionPreserved = isPrecisionPreserved(node);

    if (precisionPreserved) {
        const auto parentRestrictions = getParentInputRestrictions(node);

        // TODO: there is limitation here: one operation - one output precision
        // 1. manual merge parent inputs to one current input
        auto resultAttribute = std::make_shared<ngraph::VariantWrapper<PrecisionsAttribute>>(parentRestrictions[0]->get().sharedPart->value);
        for (size_t index = 1ul; index < parentRestrictions.size(); index++) {
            // TODO: not correct: intersection is needed here
            parentRestrictions[index]->get().sharedPart->value = resultAttribute->get().sharedPart->value;
        }

        // 2. propagate
        for (auto& input : node->inputs()) {
            auto& rt = input.get_rt_info();
            rt.emplace(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name, resultAttribute);
        }
    } else {
        // manual merge parent input separatelly
        for (Input<Node>& input : node->inputs()) {
            const auto& rt = input.get_rt_info();
            auto it = rt.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
            if (it == rt.end()) {
                continue;
            }

            const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(it->second);
            const auto& value = attribute->get().sharedPart->value;

            const auto& inputNode = input.get_source_output().get_node()->shared_from_this();
            if (isPrecisionPreserved(inputNode)) {
                for (const Input<Node>& input : inputNode->inputs()) {
                    auto& inputRtInfo = input.get_rt_info();
                    auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
                    if (inputAttributeIt != inputRtInfo.end()) {
                        const auto& inputAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(inputAttributeIt->second);
                        inputAttribute->get().sharedPart->value->precisions = value->precisions;
                    }
                }
            }

            ////const auto& inputNode = input.get_source_output().get_node()->shared_from_this();

            //const auto& parentOutput = input.get_source_output();
            //auto& parentOutputRtInfo = parentOutput.get_rt_info();
            //auto outputAttributeIt = parentOutputRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
            //if (outputAttributeIt != parentOutputRtInfo.end()) {
            //    const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(outputAttributeIt->second);
            //    parentOutputAttributes.push_back(attribute);
            //}
        }
    }


    //// propagate from parent input[s] to current one input:
    //for (Input<Node>& input : node->inputs()) {
    //    std::vector<std::shared_ptr<ngraph::VariantWrapper<PrecisionsAttribute>>> parentAttributes;
    //    const auto& inputNode = input.get_source_output().get_node()->shared_from_this();
    //    if (isPrecisionPreserved(inputNode)) {
    //        for (const Input<Node>& input : inputNode->inputs()) {
    //            auto& inputRtInfo = input.get_rt_info();
    //            auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
    //            if (inputAttributeIt != inputRtInfo.end()) {
    //                const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(inputAttributeIt->second);
    //                parentAttributes.push_back(attribute);
    //            }
    //        }
    //    } else if (is_type<opset1::FakeQuantize>(inputNode)) {
    //        const auto& outputPortRtInfo = inputNode->outputs()[0].get_rt_info();
    //        auto attributeIt = outputPortRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
    //        if (attributeIt != outputPortRtInfo.end()) {
    //            const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(attributeIt->second);
    //            parentAttributes.push_back(attribute);
    //        }
    //    }

    //    if (!parentAttributes.empty()) {
    //        // manual merge for parent inputs
    //        const auto resultAttribute = std::make_shared<ngraph::VariantWrapper<PrecisionsAttribute>>(parentAttributes[0]->get().sharedPart->value);
    //        for (size_t index = 1ul; index < parentAttributes.size(); index++) {
    //            // TODO: not correct: intersection is needed here
    //            parentAttributes[index]->get().sharedPart->value = resultAttribute->get().sharedPart->value;
    //        }

    //        if (precisionPreserved) {
    //            // propagation for current node: define inputs
    //            for (Input<Node>& input : node->inputs()) {
    //                auto& outputRtInfo = input.get_rt_info();
    //                outputRtInfo.emplace(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name, resultAttribute);
    //            }

    //            if (parentAttributes.size() >= 2ul) {
    //                // TODO: debug only
    //                const auto& value1 = parentAttributes[0]->get().sharedPart->value;
    //                const auto& value2 = parentAttributes[1]->get().sharedPart->value;
    //                const auto& value3 = resultAttribute->get().sharedPart->value;
    //            }
    //        } else {
    //            // manual merge with current input limitation
    //            const auto& rt = input.get_rt_info();
    //            const auto it = rt.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
    //            if (it != rt.end()) {
    //                // TODO: not correct: intersection is needed here
    //                const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(it->second);

    //                // TODO: debug only
    //                const auto value1 = resultAttribute->get().sharedPart->value;
    //                const auto value2 = attribute->get().sharedPart->value;

    //                resultAttribute->get().sharedPart->value = attribute->get().sharedPart->value;
    //            }
    //        }
    //    }
    //}

    //// merge
    //if (is_type<opset1::Concat>(node)) {

    //}
}

bool ngraph::pass::low_precision::PropagatePrecisions::run_on_function(std::shared_ptr<ngraph::Function> f) {
    std::vector<std::shared_ptr<ngraph::Node>> nodes(f->get_ordered_ops());
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
        const std::shared_ptr<Node> node = *it;

        // TODO: debug only
        std::cout << "PropagatePrecisions::run_on_function: " << node->get_type_name() << ": " << node->get_friendly_name() << std::endl;

        if (is_type<opset1::FakeQuantize>(node)) {
            // define
            auto& outputs = node->outputs();
            assert(outputs.size() == 1ul);
            Output<Node>& output = outputs[0];
            auto& outputRtInfo = output.get_rt_info();
            outputRtInfo.emplace(
                ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name,
                std::make_shared<ngraph::VariantWrapper<PrecisionsAttribute>>(std::set<element::Type>{element::u8, element::i8}));
            continue;

        }

        handle(node);
    }
    return true;
}

#endif
