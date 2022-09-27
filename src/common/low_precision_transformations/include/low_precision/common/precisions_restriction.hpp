// Copyright (C) 2018-2022 Intel Corporation
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

class Ports : public std::vector<size_t> {
public:
    Ports(const size_t port) : std::vector<size_t>{ port } {}
    Ports(const std::initializer_list<size_t>& ports_list) : std::vector<size_t>(ports_list) {}
};

class PrecisionsRestriction {
public:
    using PrecisionsByPorts = std::vector<std::pair<Ports, std::vector<ngraph::element::Type>>>;

    ngraph::Node::type_info_t operationType;
    bool specifyVersion;
    PrecisionsByPorts precisionsByPorts;

    PrecisionsRestriction() = default;
    PrecisionsRestriction(
        const ngraph::Node::type_info_t operationType,
        const bool specifyVersion,
        const PrecisionsByPorts& precisionsByPorts) :
        operationType(operationType),
        specifyVersion(specifyVersion),
        precisionsByPorts(precisionsByPorts) {}

    template <typename T>
    static PrecisionsRestriction create(
        const PrecisionsByPorts& precisionsByPorts,
        const bool specifyVersion = false) {
        return PrecisionsRestriction(T::get_type_info_static(), specifyVersion, precisionsByPorts);
    }

    template <typename T>
    static PrecisionsByPorts getPrecisionsByOperationType(std::vector<PrecisionsRestriction>& restrictions) {
        for (const auto& restriction : restrictions) {
            if (restriction.operationType == T::get_type_info_static()) {
                return restriction.precisionsByPorts;
            }
        }
        return {};
    }
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
