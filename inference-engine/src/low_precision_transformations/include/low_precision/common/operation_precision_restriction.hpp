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

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

//class TRANSFORMATIONS_API MarkupPrecisions;
//
//}  // namespace low_precision
//}  // namespace pass
//}  // namespace ngraph

class OperationPrecisionRestriction {
public:
    using PrecisionsByPort = std::vector<std::pair<size_t, std::set<ngraph::element::Type>>>;

    std::string name;
    int64_t version;
    std::vector<std::pair<size_t, std::set<ngraph::element::Type>>> precisionsByPort;

    OperationPrecisionRestriction() = default;
    OperationPrecisionRestriction(
        const std::string& name,
        const uint64_t version,
        const PrecisionsByPort& precisionsByPort) :
        name(name), version(version), precisionsByPort(precisionsByPort) {}

    template <typename T>
    static OperationPrecisionRestriction create(
        const PrecisionsByPort& precisionsByPort,
        const bool specifiedVersion = false) {
        const ngraph::Node::type_info_t& typeInfo = T::get_type_info_static();
        return OperationPrecisionRestriction(typeInfo.name, specifiedVersion ? typeInfo.version : -1ll, precisionsByPort);
    }
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
