// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_quantization_granularity.hpp"

#include <cassert>
#include <memory>
#include <vector>
#include "openvino/core/node.hpp"
#include "itt.hpp"
#include "low_precision/rt_info/quantization_granularity_attribute.hpp"

using namespace ov;

ov::pass::low_precision::MarkupQuantizationGranularity::MarkupQuantizationGranularity(
    const std::vector<QuantizationGranularityRestriction>& restrictions) {
    for (const auto& restriction : restrictions) {
        const auto it = restrictionsByOperation.find(restriction.operationType.name);
        if (it == restrictionsByOperation.end()) {
            PerTensorQuantization r(restriction.specifyVersion);
            r.portsByVersion.emplace(restriction.operationType.version_id, restriction.restrictions);
            restrictionsByOperation.emplace(restriction.operationType.name, r);
        } else {
            it->second.add(restriction.operationType.version_id, restriction.restrictions);
        }
    }
}

bool ov::pass::low_precision::MarkupQuantizationGranularity::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(MarkupPerTensorQuantization);
    auto setRestriction = [](const std::shared_ptr<Node>& node, const std::vector<PortQuantizationGranularityRestriction>& restrictedPorts) {
        auto createAttribute = [](Input<Node>& input, const QuantizationGranularityAttribute::Granularity& granularity){
            auto &rt = input.get_rt_info();
            rt.emplace(QuantizationGranularityAttribute::get_type_info_static(), QuantizationGranularityAttribute(granularity));
        };

        if (restrictedPorts.empty()) {
            // markup all ports with default granularity value
            for (size_t item = 0ul; item < node->get_input_size(); item++) {
                Input<Node> input = node->input(item);
                createAttribute(input, QuantizationGranularityAttribute::Granularity::PerTensor);
            }
        } else {
            // markup specific ports
            for (const auto& item : restrictedPorts) {
                Input<Node> input = node->input(item.port);
                createAttribute(input, item.granularity);
            }
        }
    };

    for (const auto& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        if (const auto multiSubGraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < multiSubGraph->get_internal_subgraphs_size(); i++)
                run_on_model(multiSubGraph->get_function(i));
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
            const auto it2 = restriction.portsByVersion.find(node->get_type_info().version_id);
            if (it2 == restriction.portsByVersion.end()) {
                continue;
            }

            const std::vector<PortQuantizationGranularityRestriction>& restrictedPorts = it2->second;
            setRestriction(node, restrictedPorts);
        } else {
            assert(restriction.portsByVersion.size() == 1ul);
            const std::vector<PortQuantizationGranularityRestriction>& restrictedPorts = restriction.portsByVersion.begin()->second;
            setRestriction(node, restrictedPorts);
        }
    }
    return true;
}
