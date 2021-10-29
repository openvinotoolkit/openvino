// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_per_tensor_quantization.hpp"

#include <cassert>
#include <memory>
#include <vector>
#include <ngraph/node.hpp>
#include "low_precision/rt_info/per_tensor_quantization_attribute.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::MarkupPerTensorQuantization, "MarkupPerTensorQuantization", 0);

ngraph::pass::low_precision::MarkupPerTensorQuantization::MarkupPerTensorQuantization(
    const std::vector<OperationPerTensorQuantizationRestriction>& restrictions) {
    for (const OperationPerTensorQuantizationRestriction& restriction : restrictions) {
        const auto it = restrictionsByOperation.find(restriction.operationType.name);
        if (it == restrictionsByOperation.end()) {
            PerTensorQuantization r(restriction.specifyVersion);
            r.portsByVersion.emplace(restriction.operationType.version, restriction.restrictedPorts);
            restrictionsByOperation.emplace(restriction.operationType.name, r);
        } else {
            it->second.add(restriction.operationType.version, restriction.restrictedPorts);
        }
    }
}

bool ngraph::pass::low_precision::MarkupPerTensorQuantization::run_on_function(std::shared_ptr<ngraph::Function> f) {
    auto setRestriction = [](const std::shared_ptr<Node>& node, const std::vector<size_t>& restrictedPorts) {
        auto createAttribute = [](Input<Node>& input){
            auto &rt = input.get_rt_info();
            rt.emplace(
                    ngraph::VariantWrapper<PerTensorQuantizationAttribute>::type_info.name,
                    std::make_shared<::ngraph::VariantWrapper<PerTensorQuantizationAttribute>>(PerTensorQuantizationAttribute()));
        };

        if (restrictedPorts.empty()) {
            // markup all ports
            for (size_t item = 0ul; item < node->get_input_size(); item++) {
                Input<Node> input = node->input(item);
                createAttribute(input);
            }
        } else {
            // markup specific ports
            for (const size_t item : restrictedPorts) {
                Input<Node> input = node->input(item);
                createAttribute(input);
            }
        }
    };

    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        const auto typeIt = restrictionsByOperation.find(node->get_type_info().name);
        if (typeIt == restrictionsByOperation.end()) {
            continue;
        }

        const auto& restriction = typeIt->second;
        if (restriction.portsByVersion.empty()) {
            continue;
        }

        if (restriction.versionIsRequired) {
            const auto it2 = restriction.portsByVersion.find(node->get_type_info().version);
            if (it2 == restriction.portsByVersion.end()) {
                continue;
            }

            const std::vector<size_t>& restrictedPorts = it2->second;
            setRestriction(node, restrictedPorts);
        } else {
            assert(restriction.portsByVersion.size() == 1ul);
            const std::vector<size_t>& restrictedPorts = restriction.portsByVersion.begin()->second;
            setRestriction(node, restrictedPorts);
        }
    }
    return true;
}
