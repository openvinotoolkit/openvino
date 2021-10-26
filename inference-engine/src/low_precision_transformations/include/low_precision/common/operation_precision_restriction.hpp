// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class OperationPrecisionRestriction {
public:
    using PrecisionsByPort = std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>>;

    ngraph::Node::type_info_t operationType;
    bool specifyVersion;
    std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>> precisionsByPort;

    OperationPrecisionRestriction() = default;
    OperationPrecisionRestriction(
        const ngraph::Node::type_info_t operationType,
        const bool specifyVersion,
        const PrecisionsByPort& precisionsByPort) :
        operationType(operationType),
        specifyVersion(specifyVersion),
        precisionsByPort(precisionsByPort) {}

    template <typename T>
    static OperationPrecisionRestriction create(
        const PrecisionsByPort& precisionsByPort,
        const bool specifyVersion = false) {
        return OperationPrecisionRestriction(T::get_type_info_static(), specifyVersion, precisionsByPort);
    }

    template <typename T>
    static PrecisionsByPort getPrecisionsByOperationType(std::vector<OperationPrecisionRestriction>& restrictions) {
        for (const auto& restriction : restrictions) {
            if (restriction.operationType == T::get_type_info_static()) {
                return restriction.precisionsByPort;
            }
        }
        return {};
    }
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
