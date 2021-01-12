// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_port_precisions.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

template class ngraph::VariantImpl<RestrictionAttribute>;

constexpr VariantTypeInfo VariantWrapper<RestrictionAttribute>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<RestrictionAttribute>::merge(const ngraph::NodeVector& nodes) {
    return nullptr;
}

std::shared_ptr<ngraph::Variant> VariantWrapper<RestrictionAttribute>::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::string getKey(const OperationRestriction& restriction) {
    if (restriction.version == -1) {
        return restriction.name;
    } else {
        return std::to_string(restriction.version) + ":" + restriction.name;
    }
}

ngraph::pass::low_precision::MarkupPortPrecisions::MarkupPortPrecisions(const std::vector<OperationRestriction>& restrictions) {
    for (const OperationRestriction& restriction : restrictions) {
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
    const std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>>& precisionsByPort) {
    for (const std::pair<size_t, std::vector<ngraph::element::Type>>& item : precisionsByPort) {
        Input<Node> input = node->input(item.first);
        auto& rt = input.get_rt_info();
        rt.emplace("precisionRestrictions", std::make_shared<::ngraph::VariantWrapper<RestrictionAttribute>>(item.second));
    }
}

bool ngraph::pass::low_precision::MarkupPortPrecisions::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
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

                const std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>> precisionsByPort = it2->second;
                setRestriction(node, precisionsByPort);
            } else {
                assert(r.precisionsByVersion.size() == 1ul);

                const std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>> precisionsByPort = r.precisionsByVersion.begin()->second;
                setRestriction(node, precisionsByPort);
            }
        }
        //Output<Node> output = node->output(0);
    }
    return true;
}
