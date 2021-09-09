// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class OperationPerTensorQuantizationRestriction {
public:
    using RestrictedPorts = std::vector<size_t>;

    ngraph::Node::type_info_t operationType;
    bool specifyVersion;
    std::vector<size_t> restrictedPorts;

    OperationPerTensorQuantizationRestriction() = default;
    OperationPerTensorQuantizationRestriction(
        const ngraph::Node::type_info_t operationType,
        const bool specifyVersion,
        const RestrictedPorts& restrictedPorts) :
        operationType(operationType),
        specifyVersion(specifyVersion),
        restrictedPorts(restrictedPorts) {}

    template <typename T>
    static OperationPerTensorQuantizationRestriction create(
        const RestrictedPorts& restrictedPorts = {},
        const bool specifyVersion = false) {
        return OperationPerTensorQuantizationRestriction(T::get_type_info_static(), specifyVersion, restrictedPorts);
    }

    template <typename T>
    static RestrictedPorts getPrecisionsByOperationType(std::vector<OperationPerTensorQuantizationRestriction>& restrictions) {
        for (const auto& restriction : restrictions) {
            if (restriction.operationType == T::get_type_info_static()) {
                return restriction.restrictedPorts;
            }
        }
        return {};
    }
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
