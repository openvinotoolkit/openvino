// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/propagate_precisions.hpp"

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

bool isPrecisionPreserved(std::shared_ptr<Node> node) {
    auto& rtInfo = node->get_rt_info();
    auto it = rtInfo.find(ngraph::VariantWrapper<PrecisionPreservedAttribute>::type_info.name);
    if (it == rtInfo.end()) {
        return false;
    }

    auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionPreservedAttribute>>(it->second);
    return attribute->get().sharedValue->value;
}

//void handle(const std::shared_ptr<ngraph::Node>& node) {
//    bool outputRestrictionsInitialized = false;
//    std::set<ngraph::element::Type> outputRestrictions;
//
//    for (Output<Node>& output : node->outputs()) {
//        //bool outputRestrictionsInitialized = false;
//
//        const auto& inputs = output.get_target_inputs();
//        for (const Input<Node>& input : inputs) {
//            auto& inputRtInfo = input.get_rt_info();
//            auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
//            if (inputAttributeIt != inputRtInfo.end()) {
//                std::shared_ptr<ngraph::VariantWrapper<PrecisionsAttribute>> inputPrecisionsAttribute =
//                    std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(inputAttributeIt->second);
//
//                // TODO: variant #1: merge attributes: not abvious how interpret `nodes`
//                // virtual std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector & nodes);
//                //inputPrecisionsAttribute->merge();
//
//                // TODO: variant #2: merge manually
//                // TODO: need tests
//                std::set<ngraph::element::Type> inputPrecisions = inputPrecisionsAttribute->get();
//                if (inputPrecisions.empty()) {
//                    continue;
//                }
//                if (outputRestrictionsInitialized) {
//                    auto it = outputRestrictions.begin();
//                    while (it != outputRestrictions.end()) {
//                        if (inputPrecisions.find(*it) == inputPrecisions.end()) {
//                            auto itNext = it;
//                            itNext++;
//                            outputRestrictions.erase(it);
//                            it = itNext;
//                        } else {
//                            it++;
//                        }
//                    }
//
//                } else {
//                    outputRestrictionsInitialized = true;
//                    outputRestrictions.insert(inputPrecisions.begin(), inputPrecisions.end());
//                }
//            }
//        }
//
//        //if (outputRestrictionsInitialized) {
//        //    auto& outputRtInfo = output.get_rt_info();
//        //    outputRtInfo.emplace(
//        //        ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name,
//        //        std::make_shared<::ngraph::VariantWrapper<PrecisionsAttribute>>(outputRestrictions));
//        //}
//    }
//
//    if (outputRestrictionsInitialized) {
//        if (is_type<opset1::FakeQuantize>(node)) {
//            auto& outputs = node->outputs();
//            Output<Node>& output = outputs[0];
//            auto& outputRtInfo = output.get_rt_info();
//            outputRtInfo.emplace(
//                ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name,
//                std::make_shared<::ngraph::VariantWrapper<PrecisionsAttribute>>(outputRestrictions));
//        } else {
//            for (Input<Node>& input : node->inputs()) {
//                auto& outputRtInfo = input.get_rt_info();
//                outputRtInfo.emplace(
//                    ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name,
//                    std::make_shared<::ngraph::VariantWrapper<PrecisionsAttribute>>(outputRestrictions));
//            }
//        }
//    }
//}

//bool ngraph::pass::low_precision::PropagatePrecisions::run_on_function(std::shared_ptr<ngraph::Function> f) {
//    std::deque<std::shared_ptr<Node>> nodes;
//    std::set<std::shared_ptr<Node>> visited;
//    for (auto& r : f->get_results()) {
//        nodes.push_back(r);
//    }
//
//    for (auto& r : f->get_sinks()) {
//        nodes.emplace_back(r);
//    }
//
//    while (!nodes.empty()) {
//        auto curr_node = nodes.front();
//        nodes.pop_front();
//
//        if (visited.count(curr_node) || is_type<op::Constant>(curr_node)) {
//            continue;
//        }
//
//        visited.insert(curr_node);
//
//        std::cout << "PropagatePrecisions::run_on_function: " << curr_node->get_type_name() << ": " << curr_node->get_friendly_name() << std::endl;
//        if (is_type<opset1::FakeQuantize>(curr_node) || isPrecisionPreserved(curr_node)) {
//            handle(curr_node);
//        }
//
//        for (auto& input_value : curr_node->input_values()) {
//            const auto& input_node = input_value.get_node_shared_ptr();
//            nodes.push_front(input_node);
//        }
//    }
//    return true;
//}

void handle(const std::shared_ptr<ngraph::Node>& node) {
    bool outputRestrictionsInitialized = false;
    std::set<ngraph::element::Type> outputRestrictions;

    for (Output<Node>& output : node->outputs()) {
        //bool outputRestrictionsInitialized = false;

        const auto& inputs = output.get_target_inputs();
        for (const Input<Node>& input : inputs) {
            auto& inputRtInfo = input.get_rt_info();
            auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
            //if (inputAttributeIt != inputRtInfo.end()) {
            //    std::shared_ptr<ngraph::VariantWrapper<PrecisionsAttribute>> inputPrecisionsAttribute =
            //        std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(inputAttributeIt->second);

            //    // TODO: variant #1: merge attributes: not abvious how interpret `nodes`
            //    // virtual std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector & nodes);
            //    //inputPrecisionsAttribute->merge();

            //    // TODO: variant #2: merge manually
            //    // TODO: need tests
            //    std::set<ngraph::element::Type> inputPrecisions = inputPrecisionsAttribute->get();
            //    if (inputPrecisions.empty()) {
            //        continue;
            //    }
            //    if (outputRestrictionsInitialized) {
            //        auto it = outputRestrictions.begin();
            //        while (it != outputRestrictions.end()) {
            //            if (inputPrecisions.find(*it) == inputPrecisions.end()) {
            //                auto itNext = it;
            //                itNext++;
            //                outputRestrictions.erase(it);
            //                it = itNext;
            //            } else {
            //                it++;
            //            }
            //        }

            //    } else {
            //        outputRestrictionsInitialized = true;
            //        outputRestrictions.insert(inputPrecisions.begin(), inputPrecisions.end());
            //    }
            //}
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
