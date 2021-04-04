// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_precisions.hpp"

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

ngraph::pass::low_precision::MarkupPrecisions::MarkupPrecisions(const std::vector<OperationPrecisionRestriction>& restrictions) {
    for (const OperationPrecisionRestriction& restriction : restrictions) {
        const auto it = restrictionsByOperation.find(restriction.name);
        if (it == restrictionsByOperation.end()) {
            Restriction r(restriction.version != -1);
            r.precisionsByVersion.emplace(restriction.version, restriction.precisionsByPort);
            restrictionsByOperation.emplace(restriction.name, r);
        } else {
            it->second.add(restriction.version, restriction.precisionsByPort);
        }
    }
}

void setRestriction(
    const std::shared_ptr<Node>& node,
    const std::vector<std::pair<size_t, std::set<ngraph::element::Type>>>& precisionsByPort) {
    if (precisionsByPort.empty()) {
        // if available precisions for any port is empty then mark all input ports
        for (auto& input : node->inputs()) {
            auto& rt = input.get_rt_info();
            rt.emplace(
                ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name,
                std::make_shared<::ngraph::VariantWrapper<PrecisionsAttribute>>(PrecisionsAttribute(std::set<element::Type>())));
        }
    } else {
        for (const std::pair<size_t, std::set<ngraph::element::Type>>& item : precisionsByPort) {
            Input<Node> input = node->input(item.first);
            auto& rt = input.get_rt_info();

            // if available precisions for any port is empty then don't update anything
            const auto it = rt.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
            if (it != rt.end()) {
                auto precisionsAttribute = std::dynamic_pointer_cast<PrecisionsAttribute>((*it).second);
                if (precisionsAttribute->sharedPart->value->precisions.empty()) {
                    return;
                }
            }
            rt.emplace(
                ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name,
                std::make_shared<::ngraph::VariantWrapper<PrecisionsAttribute>>(item.second));
        }
    }
}

bool ngraph::pass::low_precision::MarkupPrecisions::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        const bool precisionPreserved = isPrecisionPreserved(node);
        if (precisionPreserved) {
            auto& rt = node->get_rt_info();
            rt.emplace(
                ngraph::VariantWrapper<PrecisionPreservedAttribute>::type_info.name,
                std::make_shared<::ngraph::VariantWrapper<PrecisionPreservedAttribute>>(precisionPreserved));
        }

        const auto& typeInfo = node->get_type_info();
        auto it = restrictionsByOperation.find(typeInfo.name);
        if (it != restrictionsByOperation.end()) {
            const Restriction& r = it->second;
            if (r.versionIsRequired) {
                const auto it2 = r.precisionsByVersion.find(typeInfo.version);
                if (it2 == r.precisionsByVersion.end()) {
                    continue;
                }

                const std::vector<std::pair<size_t, std::set<ngraph::element::Type>>> precisionsByPort = it2->second;
                setRestriction(node, precisionsByPort);
            } else {
                assert(r.precisionsByVersion.size() == 1ul);

                const std::vector<std::pair<size_t, std::set<ngraph::element::Type>>> precisionsByPort = r.precisionsByVersion.begin()->second;
                setRestriction(node, precisionsByPort);
            }
        }


        //Output<Node> output = node->output(0);
    }
    return true;
}

bool ngraph::pass::low_precision::MarkupPrecisions::isDisabled(const std::shared_ptr<Node>& node) {
    for (const auto& input : node->inputs()) {
        auto& rtInfo = input.get_rt_info();
        auto it = rtInfo.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
        if (it == rtInfo.end()) {
            continue;
        }

        auto precisionAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(it->second);
        assert(precisionAttribute != nullptr);
        const std::set<ngraph::element::Type>& precisionRestrictions = precisionAttribute->get().sharedPart->value->precisions;
        if (precisionRestrictions.empty()) {
            return true;
        }
    }
    return false;
}

bool ngraph::pass::low_precision::MarkupPrecisions::isPrecisionPreserved(const std::shared_ptr<Node>& node) {
    if (isDisabled(node)) {
        return false;
    }

    static std::unordered_set<std::string> precisionPreserved = {
        { "Concat" },
        { "MaxPool" }
    };

    return precisionPreserved.find(node->get_type_name()) != precisionPreserved.end();
}

bool ngraph::pass::low_precision::MarkupPrecisions::isQuantized(const std::shared_ptr<Node>& node) {
    return true;
}
