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
        const auto it = restrictionsByOperation.find(restriction.operationType.name);
        if (it == restrictionsByOperation.end()) {
            Restriction r(restriction.specifyVersion);
            r.precisionsByVersion.emplace(restriction.operationType.version, restriction.precisionsByPort);
            restrictionsByOperation.emplace(restriction.operationType.name, r);
        } else {
            it->second.add(restriction.operationType.version, restriction.precisionsByPort);
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

            auto attribute = ngraph::pass::low_precision::make_shared_attribute<PrecisionsAttribute>(std::set<element::Type>());
            auto attributeWrapper = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attribute);

            rt.emplace(
                ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name,
                attributeWrapper);
        }
    } else {
        for (const std::pair<size_t, std::set<ngraph::element::Type>>& item : precisionsByPort) {
            Input<Node> input = node->input(item.first);
            auto& rt = input.get_rt_info();

            // if available precisions for any port is empty then don't update anything
            const auto it = rt.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
            if (it != rt.end()) {
                auto var = (*it).second;
                auto precisionsAttribute = std::dynamic_pointer_cast<PrecisionsAttribute>(var);
                if (precisionsAttribute->sharedValue->precisions.empty()) {
                    return;
                }
            }

            auto attribute = ngraph::pass::low_precision::make_shared_attribute<PrecisionsAttribute>(item.second);
            auto attributeWrapper = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attribute);

            rt[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = attributeWrapper;
        }
    }
}

bool ngraph::pass::low_precision::MarkupPrecisions::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        // TODO: move outside
        const bool precisionPreserved = isPrecisionPreserved(node);
        if (precisionPreserved) {
            auto& rt = node->get_rt_info();
            rt.emplace(
                ngraph::VariantWrapper<PrecisionPreservedAttributePtr>::type_info.name,
                std::make_shared<::ngraph::VariantWrapper<PrecisionPreservedAttributePtr>>(
                    make_shared_attribute<PrecisionPreservedAttribute>(precisionPreserved)));
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
    }
    return true;
}

bool ngraph::pass::low_precision::MarkupPrecisions::isDisabled(const std::shared_ptr<Node>& node) {
    for (const auto& input : node->inputs()) {
        auto& rtInfo = input.get_rt_info();
        auto it = rtInfo.find(ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name);
        if (it == rtInfo.end()) {
            continue;
        }

        auto precisionAttribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(it->second);
        assert(precisionAttribute != nullptr);
        const std::set<ngraph::element::Type>& precisionRestrictions = precisionAttribute->get()->sharedValue->precisions;
        if (precisionRestrictions.empty()) {
            return true;
        }
    }
    return false;
}

template <class Operation>
std::string name() {
    return Operation::get_type_info_static().name;
}

bool ngraph::pass::low_precision::MarkupPrecisions::isPrecisionPreserved(const std::shared_ptr<Node>& node) {
    if (isDisabled(node)) {
        return false;
    }

    // TODO: think how to handle conditions <= not mandatory for PoC
    // TODO: operation set version is not affected <= not mandatory for PoC
    static std::unordered_set<std::string> precisionPreserved = {
        { name<opset1::Concat>() },
        { name<opset1::DepthToSpace>() },
        { name<opset1::MaxPool>() },
        { name<opset1::Relu>() },
        // TODO: there are conditions
        { name<opset1::Reshape>() },
        { name<opset1::Squeeze>() },
        { name<opset1::Split>() },
        { name<opset1::StridedSlice>() },
        { name<opset1::Transpose>() },
        { name<opset1::Unsqueeze>() },
        { name<opset1::VariadicSplit>() }
    };

    return precisionPreserved.find(node->get_type_name()) != precisionPreserved.end();
}

bool ngraph::pass::low_precision::MarkupPrecisions::isQuantized(const std::shared_ptr<Node>& node) {
    return true;
}
