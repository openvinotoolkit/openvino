// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "low_precision/lpt_visibility.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace pass {
namespace low_precision {
/**
* @brief PrecisionsRestriction defines a set of precision restrictions for each input port
* Common precision restriction can be also set for several ports. In this case, an operation will have
* the same precision for mentioned
*
* // One restriction for each port
* PrecisionsRestriction::create<ov::opset1::Convolution>({
*      {{0}, {ov::element::u8}},
*      {{1}, {ov::element::i8}},
*  }),
*
* // Common precision restriction for several ports:
* // both inputs will have the same precision
* PrecisionsRestriction::create<ov::opset5::LSTMSequence>({
*      {{0, 1}, {ov::element::u8, ov::element::i8}}
*  }),
*/
class PrecisionsRestriction {
public:
    using PrecisionsByPorts = std::vector<std::pair<std::vector<size_t>, std::vector<ov::element::Type>>>;
    using PrecisionsByPortsFunction = std::function<PrecisionsByPorts(const std::shared_ptr<Node>&)>;

    ov::Node::type_info_t operationType;
    bool specifyVersion;
    PrecisionsByPorts precisionsByPorts;
    PrecisionsByPortsFunction precisionsByPortsFunction;

    PrecisionsRestriction() = default;
    PrecisionsRestriction(
        const ov::Node::type_info_t& operationType,
        const bool specifyVersion,
        const PrecisionsByPorts& precisionsByPorts) :
        operationType(operationType),
        specifyVersion(specifyVersion),
        precisionsByPorts(precisionsByPorts) {}

    PrecisionsRestriction(
        const ov::Node::type_info_t& operationType,
        const bool specifyVersion,
        const PrecisionsByPortsFunction& precisionsByPortsFunction) :
        operationType(operationType),
        specifyVersion(specifyVersion),
        precisionsByPortsFunction(precisionsByPortsFunction) {}

    template <typename T>
    static PrecisionsRestriction create(
        const PrecisionsByPorts& precisionsByPorts,
        const bool specifyVersion = false) {
        return PrecisionsRestriction(T::get_type_info_static(), specifyVersion, precisionsByPorts);
    }

    template <typename T>
    static PrecisionsRestriction create(
        const PrecisionsByPortsFunction& precisionsByPortsFunction,
        const bool specifyVersion = false) {
        return PrecisionsRestriction(T::get_type_info_static(), specifyVersion, precisionsByPortsFunction);
    }

    template <typename T>
    static PrecisionsByPorts getPrecisionsByOperationType(const std::vector<PrecisionsRestriction>& restrictions) {
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
}  // namespace ov
