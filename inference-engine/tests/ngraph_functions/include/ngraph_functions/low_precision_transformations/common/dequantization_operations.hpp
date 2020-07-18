// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/layer_transformation.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class DequantizationOperations {
public:
    DequantizationOperations();

    DequantizationOperations(
        const ngraph::element::Type_t convertOutputPrecision,
        const std::vector<float>& subtractValues,
        const std::vector<float>& multiplyValues);

    const ngraph::element::Type_t convertOutputPrecision;
    const std::vector<float> subtractValues;
    const std::vector<float> multiplyValues;
};

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

inline std::ostream& operator<<(std::ostream& out, const DequantizationOperations& data) {
    return out << "_" << data.convertOutputPrecision << "_" << data.subtractValues << "_" << data.multiplyValues;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
