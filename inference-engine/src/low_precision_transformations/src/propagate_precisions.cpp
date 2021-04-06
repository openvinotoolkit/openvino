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
#include "low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

// 0, 1 - backward propagation, restriction operations begin
// 2 - forward propagation, FakeQuantize operations begin
#define TYPE 2

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

std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> getParentInputRestrictions(const std::shared_ptr<ngraph::Node> node) {
    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> parentAttributes;
    for (Input<Node>& input : node->inputs()) {
        const auto& inputNode = input.get_source_output().get_node()->shared_from_this();
        if (NetworkHelper::isPrecisionPreserved(inputNode)) {
            for (const Input<Node>& input : inputNode->inputs()) {
                auto& inputRtInfo = input.get_rt_info();
                auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
                if (inputAttributeIt != inputRtInfo.end()) {
                    const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(inputAttributeIt->second);
                    parentAttributes.push_back(attribute);
                }
            }
        } else if (is_type<opset1::FakeQuantize>(inputNode)) {
            const auto& outputPortRtInfo = inputNode->outputs()[0].get_rt_info();
            auto attributeIt = outputPortRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
            if (attributeIt != outputPortRtInfo.end()) {
                const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attributeIt->second);
                parentAttributes.push_back(attribute);
            }
        }
    }
    return parentAttributes;
}

std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> getChildrenInputRestrictions(const std::shared_ptr<ngraph::Node> node) {
    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> childAttributes;
    for (Output<Node>& output : node->outputs()) {
        for (const Input<Node>& input : output.get_target_inputs()) {
            auto& inputRtInfo = input.get_rt_info();
            auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
            if (inputAttributeIt != inputRtInfo.end()) {
                const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(inputAttributeIt->second);
                childAttributes.push_back(attribute);
            }
        }
    }
    return childAttributes;
}

std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>> getParentInputRestriction(const std::shared_ptr<ngraph::Node> node, const size_t parentIndex) {
    const auto& inputs = node->inputs();
    Input<Node> input = inputs[parentIndex];
    const auto& inputNode = input.get_source_output().get_node()->shared_from_this();
    if (NetworkHelper::isPrecisionPreserved(inputNode)) {
        for (const Input<Node>& input : inputNode->inputs()) {
            auto& inputRtInfo = input.get_rt_info();
            auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
            if (inputAttributeIt != inputRtInfo.end()) {
                const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(inputAttributeIt->second);
                return attribute;
            }
        }
    }

    if (is_type<opset1::FakeQuantize>(inputNode)) {
        const auto& outputPortRtInfo = inputNode->outputs()[0].get_rt_info();
        auto attributeIt = outputPortRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
        if (attributeIt != outputPortRtInfo.end()) {
            const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attributeIt->second);
            return attribute;
        }
    }

    return nullptr;
}


std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> getParentOutputRestrictions(const std::shared_ptr<ngraph::Node> node) {
    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> parentOutputAttributes;
    for (Input<Node>& input : node->inputs()) {
        const auto& inputNode = input.get_source_output().get_node()->shared_from_this();
        const auto& parentOutput = input.get_source_output();

        auto& parentOutputRtInfo = parentOutput.get_rt_info();
        auto outputAttributeIt = parentOutputRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
        if (outputAttributeIt != parentOutputRtInfo.end()) {
            const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(outputAttributeIt->second);
            parentOutputAttributes.push_back(attribute);
        }
    }
    return parentOutputAttributes;
}

void replaceAttributeInInputs(
    std::shared_ptr<ngraph::Function> f,
    const std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>> newAttribute,
    const std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>> oldAttribute,
    const std::shared_ptr<ngraph::Node>& initialNode) {
    const std::string name = ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name;

    std::set<std::shared_ptr<Node>> visited;
    std::deque<std::shared_ptr<Node>> nodes;
    nodes.emplace_back(initialNode);

    bool initialNodeIsNotInitialized = true;

    while (!nodes.empty()) {
        auto node = nodes.front();
        nodes.pop_front();

        if (visited.count(node) || is_type<op::Constant>(node)) {
            continue;
        }

        visited.insert(node);

        bool handleConnectedNodes = false;
        if (is_type<opset1::FakeQuantize>(node)) {
            for (auto& output : node->outputs()) {
                auto& rt = output.get_rt_info();
                if (node == initialNode) {
                    rt[name] = newAttribute;
                    handleConnectedNodes = true;
                } else {
                    auto it = rt.find(name);
                    if (it != rt.end()) {
                        const auto currentAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
                        const ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>* raw1 = oldAttribute.get();
                        const ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>* raw2 = currentAttribute.get();
                        if (raw1 == raw2) {
                            rt[name] = newAttribute;
                        }
                        handleConnectedNodes = true;
                    }
                }
            }
        } else {
            for (auto& input : node->inputs()) {
                auto& rt = input.get_rt_info();

                if (node == initialNode) {
                    rt[name] = newAttribute;
                    handleConnectedNodes = true;
                } else {
                    auto it = rt.find(name);
                    if (it != rt.end()) {
                        const auto currentAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
                        const ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>* raw1 = oldAttribute.get();
                        const ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>* raw2 = currentAttribute.get();
                        if (raw1 == raw2) {
                            rt[name] = newAttribute;
                        }
                        handleConnectedNodes = true;
                    }
                }
            }
        }

        if (!handleConnectedNodes) {
            continue;
        }

        if (!is_type<opset1::FakeQuantize>(node)) {
            for (auto& input : node->inputs()) {
                const auto& input_node = input.get_source_output().get_node_shared_ptr();
                if (visited.count(input_node) || is_type<op::Constant>(input_node)) {
                    continue;
                }

                nodes.push_front(input_node);
            }
        }

        for (auto& output : node->outputs()) {
            for (auto& input_value : output.get_target_inputs()) {
                const auto& output_node = input_value.get_node()->shared_from_this();
                if (visited.count(output_node) || is_type<op::Constant>(output_node)) {
                    continue;
                }

                nodes.push_front(output_node);
            }
        }
    }
}

void handle(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::Node>& node) {
    // TODO: possible need to add validation here to avoid not neccaassary actions for not preserved operations without precision limitations
    const bool precisionPreserved = NetworkHelper::isPrecisionPreserved(node);

    if (precisionPreserved) {
        const auto parentRestrictions = getParentInputRestrictions(node);
        //const auto parentRestrictions = getChildrenInputRestrictions(node);
        if (parentRestrictions.empty()) {
            return;
        }

        // TODO: there is limitation here: one operation - one output precision
        // 1. manual merge parent inputs to one current output

        //auto attribute = std::make_shared<PrecisionsAttribute>(parentRestrictions[0]->get()->sharedPart);
        //auto resultAttribute = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attribute);

        auto resultAttribute = parentRestrictions[0];

        std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> toMerge = parentRestrictions;
        toMerge.erase(toMerge.begin());
        resultAttribute->merge(toMerge);

        //auto resultAttribute = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(parentRestrictions[0]->get()->sharedPart->value);

        for (size_t index = 1ul; index < parentRestrictions.size(); index++) {
            //const auto newAttribute = resultAttribute->get();
            //const auto oldAttribute = parentRestrictions[index]->get();

            //for (auto& output : node->inputs()) {
            //    auto& nodeRt = output.get_rt_info();
            //    nodeRt[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = oldAttribute;
            //}

            //replace(f, newAttribute, oldAttribute, node);

            // TODO: not correct: intersection is needed here
            // parentRestrictions[index]->get()->sharedPart = resultAttribute->get()->sharedPart;

            const auto oldAttribute = parentRestrictions[index]->get();

            //for (auto& input : node->inputs()) {
            //    auto& rt = input.get_rt_info();
            //    rt[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = parentRestrictions[index];
            //}

            replaceAttributeInInputs(f, resultAttribute, parentRestrictions[index], node);
        }

        if (is_type<opset1::FakeQuantize>(node)) {
            auto& outputPortRtInfo = node->outputs()[0].get_rt_info();
            //auto attributeIt = outputPortRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
            //if (attributeIt != outputPortRtInfo.end()) {
            //    const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(attributeIt->second);
            //    childAttributes.push_back(attribute);
            //}
            outputPortRtInfo[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = resultAttribute;
        } else {
            // 2. propagate
            for (auto& input : node->inputs()) {
                auto& rt = input.get_rt_info();
                rt[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = resultAttribute;
            }
        }
    }
}

bool ngraph::pass::low_precision::PropagatePrecisions::run_on_function(std::shared_ptr<ngraph::Function> f) {
    std::vector<std::shared_ptr<ngraph::Node>> nodes(f->get_ordered_ops());
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
        const std::shared_ptr<Node> node = *it;

        // TODO: debug only
        std::cout << "PropagatePrecisions::run_on_function: " << node->get_type_name() << ": " << node->get_friendly_name() << ": BEGIN" << std::endl;

        if (is_type<opset1::FakeQuantize>(node)) {
            // define
            auto& outputs = node->outputs();
            assert(outputs.size() == 1ul);
            Output<Node>& output = outputs[0];
            auto& outputRtInfo = output.get_rt_info();

            auto attribute = std::make_shared<PrecisionsAttribute>(std::set<element::Type>{element::u8, element::i8});
            auto attributeWrapper = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attribute);
            outputRtInfo[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = attributeWrapper;
            continue;
        }

        if (!NetworkHelper::isPrecisionPreserved(node)) {
            for (auto& input : node->inputs()) {
                auto parentNode = input.get_source_output().get_node_shared_ptr();

                // TODO: move to method
                auto getAttributes = [](const Input<Node>& nodeInput) {
                    const static std::string name = ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name;

                    auto node = nodeInput.get_source_output().get_node_shared_ptr();
                    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> attributes;
                    if (is_type<opset1::FakeQuantize>(node)) {
                        // output
                        auto& rt = nodeInput.get_source_output().get_rt_info();
                        auto it = rt.find(name);
                        if (it != rt.end()) {
                            const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
                            attributes.push_back(attribute);
                        }
                    } else if (NetworkHelper::isPrecisionPreserved(node)) {
                        // inputs
                        for (auto input : node->inputs()) {
                            auto& rt = input.get_rt_info();
                            auto it = rt.find(name);
                            if (it == rt.end()) {
                                continue;
                            }
                            const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
                            attributes.push_back(attribute);
                        }
                    }

                    return attributes;
                };

                auto parentAttributes = getAttributes(input);
                if (parentAttributes.empty()) {
                    continue;
                }

                auto& nodeRt = input.get_rt_info();

                const static std::string name = ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name;
                const auto it = nodeRt.find(name);
                if (it == nodeRt.end()) {
                    continue;
                }

                const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
                std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> attributes{ attribute};

                for (auto& parentAttribute : parentAttributes) {
                    parentAttribute->merge(attributes);
                    //auto precisions1 = parentAttribute->get()->sharedPart->value->precisions;
                    //auto precisions2 = attributes[0]->get()->sharedPart->value->precisions;
                }

                nodeRt[name] = parentAttributes[0];
            }
            //continue;
        }

        if (NetworkHelper::isPrecisionPreserved(node)) {
            handle(f, node);
        }

        std::cout << "PropagatePrecisions::run_on_function: " << node->get_type_name() << ": " << node->get_friendly_name() << ": END" << std::endl;
        for (auto& input : node->inputs()) {
            const std::string name = ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name;
            auto& nodeRt = input.get_rt_info();
            const auto it = nodeRt.find(name);
            if (it == nodeRt.end()) {
                continue;
            }

            const auto& attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
            std::cout << name << ": " << attribute->get_string() << std::endl;
        }
    }
    return true;
}

#endif
