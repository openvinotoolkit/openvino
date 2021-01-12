// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/propagate_shared_value.hpp"

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

namespace ngraph {
namespace pass {
namespace low_precision {

//std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> PropagateSharedValue::getParentInputRestrictions(
//    const std::shared_ptr<ngraph::Node> node) {
//    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> parentAttributes;
//    for (size_t index = 0ul; index < node->get_input_size(); index++) {
//        const Input<Node>& input = node->input(index);
//        auto inputNode = input.get_source_output().get_node()->shared_from_this();
//
//        const auto dequantization = NetworkHelper::getDequantization(node, index);
//        if (!dequantization.empty() &&
//            (is_type<opset1::Convert>(dequantization.data.get_node())) &&
//            is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
//            inputNode = dequantization.data.get_node()->get_input_node_shared_ptr(0);
//        }
//
//        if (NetworkHelper::isPrecisionPreserved(inputNode)) {
//            //for (const Input<Node>& input : inputNode->inputs()) {
//            //    auto& inputRtInfo = input.get_rt_info();
//            //    auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
//            //    if (inputAttributeIt != inputRtInfo.end()) {
//            //        const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(
//            //            inputAttributeIt->second);
//            //        parentAttributes.push_back(attribute);
//            //    }
//            //}
//
//            auto& inputRtInfo = inputNode->get_rt_info();
//            auto inputAttributeIt = inputRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
//            if (inputAttributeIt != inputRtInfo.end()) {
//                const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(inputAttributeIt->second);
//                parentAttributes.push_back(attribute);
//            }
//        } else if (is_type<opset1::FakeQuantize>(inputNode)) {
//            const auto& outputPortRtInfo = inputNode->outputs()[0].get_rt_info();
//            auto attributeIt = outputPortRtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
//            if (attributeIt != outputPortRtInfo.end()) {
//                const auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attributeIt->second);
//                parentAttributes.push_back(attribute);
//            }
//        }
//    }
//    return parentAttributes;
//}
//
//void PropagateSharedValue::handle(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::Node>& node) {
//    // TODO: possible need to add validation here to avoid not neccaassary actions for not preserved operations without precision limitations
//    const bool precisionPreserved = NetworkHelper::isPrecisionPreserved(node);
//
//    if (precisionPreserved) {
//        const auto parentRestrictions = getParentInputRestrictions(node);
//        if (parentRestrictions.empty()) {
//            return;
//        }
//
//        // TODO: there is limitation here: one operation - one output precision
//        // 1. merge parent inputs to one current output
//        auto resultAttribute = parentRestrictions[0];
//
//        std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>> toMerge = parentRestrictions;
//        toMerge.erase(toMerge.begin());
//        resultAttribute->merge(toMerge);
//
//        for (size_t index = 1ul; index < parentRestrictions.size(); index++) {
//            const auto oldAttribute = parentRestrictions[index]->get();
//            //replaceAttributeInInputs(f, resultAttribute, parentRestrictions[index], node);
//
//            NetworkHelper::reassign<PrecisionsSharedValue, PrecisionsAttribute>(
//                resultAttribute->get()->sharedValue,
//                parentRestrictions[index]->get()->sharedValue->attributes);
//        }
//
//        auto& rt = node->get_rt_info();
//        rt[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = resultAttribute;
//
//        //// 2. propagate
//        //if (is_type<opset1::FakeQuantize>(node)) {
//        //    auto& outputPortRtInfo = node->outputs()[0].get_rt_info();
//        //    outputPortRtInfo[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = resultAttribute;
//        //} else {
//        //    for (auto& input : node->inputs()) {
//        //        auto& rt = input.get_rt_info();
//        //        rt[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = resultAttribute;
//        //    }
//        //}
//    }
//}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
