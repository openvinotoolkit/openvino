// Copyright (C) 2018-2023 Intel Corporation
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
/**
* @brief PrecisionsRestriction defines a set of precision restrictions for each input port
* Common precision restriction can be also set for several ports. In this case, an operation will have
* the same precision for mentioned
*
* // One restriction for each port
* PrecisionsRestriction::create<ngraph::opset1::Convolution>({
*      {{0}, {ngraph::element::u8}},
*      {{1}, {ngraph::element::i8}},
*  }),
*
* // Common precision restriction for several ports:
* // both inputs will have the same precision
* PrecisionsRestriction::create<ngraph::opset5::LSTMSequence>({
*      {{0, 1}, {ngraph::element::u8, ngraph::element::i8}}
*  }),
*/
class PrecisionsRestriction {
public:
    using PrecisionsByPorts = std::vector<std::pair<std::vector<size_t>, std::vector<ngraph::element::Type>>>;

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
