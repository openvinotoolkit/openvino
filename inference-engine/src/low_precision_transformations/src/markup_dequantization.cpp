// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_dequantization.hpp"

#include <memory>
#include <unordered_set>
#include <set>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

using namespace ngraph;

//void setRestriction(
//    const std::shared_ptr<Node>& node,
//    const std::vector<std::pair<size_t, std::set<ngraph::element::Type>>>& precisionsByPort) {
//    if (precisionsByPort.empty()) {
//        // if available precisions for any port is empty then mark all input ports
//        for (auto& input : node->inputs()) {
//            auto& rt = input.get_rt_info();
//
//            auto attribute = std::make_shared<PrecisionsAttribute>(std::set<element::Type>());
//            auto attributeWrapper = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attribute);
//
//            rt.emplace(
//                ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name,
//                attributeWrapper);
//        }
//    } else {
//        for (const std::pair<size_t, std::set<ngraph::element::Type>>& item : precisionsByPort) {
//            Input<Node> input = node->input(item.first);
//            auto& rt = input.get_rt_info();
//
//            // if available precisions for any port is empty then don't update anything
//            const auto it = rt.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
//            if (it != rt.end()) {
//                auto var = (*it).second;
//                auto precisionsAttribute = std::dynamic_pointer_cast<PrecisionsAttribute>(var);
//                if (precisionsAttribute->precisions.empty()) {
//                    return;
//                }
//            }
//
//            auto attribute = std::make_shared<PrecisionsAttribute>(item.second);
//            auto attributeWrapper = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attribute);
//
//            rt.emplace(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name, attributeWrapper);
//        }
//    }
//}

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::MarkupDequantizations, "MarkupDequantizations", 0);

ngraph::pass::low_precision::MarkupDequantizations::MarkupDequantizations() {
    auto matcherData = ngraph::pattern::any_input();
    auto matcherConvert = ngraph::pattern::wrap_type<opset3::Convert>({ matcherData }, pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) -> bool {
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcherConvert, "MarkupDequantizations");
    this->register_matcher(m, callback);
}
