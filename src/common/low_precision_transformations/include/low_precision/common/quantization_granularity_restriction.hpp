// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/node.hpp"

#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/quantization_granularity_attribute.hpp"
#include "low_precision/common/port_quantization_granularity_restriction.hpp"


namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API QuantizationGranularityRestriction {
public:
    ov::Node::type_info_t operationType;
    bool specifyVersion;
    std::vector<PortQuantizationGranularityRestriction> restrictions;

    QuantizationGranularityRestriction() = default;
    QuantizationGranularityRestriction(
        const ov::Node::type_info_t operationType,
        const bool specifyVersion,
        const std::vector<PortQuantizationGranularityRestriction>& restrictions) :
        operationType(operationType),
        specifyVersion(specifyVersion),
        restrictions(restrictions) {}

    template <typename T>
    static QuantizationGranularityRestriction create(
        const std::vector<PortQuantizationGranularityRestriction>& restrictions,
        const bool specifyVersion) {
        return QuantizationGranularityRestriction(T::get_type_info_static(), specifyVersion, restrictions);
    }

    template <typename T>
    static QuantizationGranularityRestriction create(
        const std::vector<size_t>& restrictedPorts = {},
        const bool specifyVersion = false) {
        std::vector<PortQuantizationGranularityRestriction> restrictions;
        restrictions.reserve(restrictedPorts.size());
        for (auto i = 0ul; i < restrictedPorts.size(); ++i) {
            restrictions.push_back(PortQuantizationGranularityRestriction(
                restrictedPorts[i],
                ov::QuantizationGranularityAttribute::Granularity::PerTensor));
        }
        return QuantizationGranularityRestriction(T::get_type_info_static(), specifyVersion, restrictions);
    }

    template <typename T>
    static std::vector<PortQuantizationGranularityRestriction> getPrecisionsByOperationType(std::vector<QuantizationGranularityRestriction>& restrictions) {
        for (const auto& restriction : restrictions) {
            if (restriction.operationType == T::get_type_info_static()) {
                return restriction.restrictions;
            }
        }
        return {};
    }
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
