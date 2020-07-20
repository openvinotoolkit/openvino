// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "fake_quantize_on_data.hpp"

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

    bool empty() const;

    ngraph::element::Type_t convertOutputPrecision;
    std::vector<float> subtractValues;
    std::vector<float> multiplyValues;
};

inline std::ostream& operator<<(std::ostream& out, const DequantizationOperations& data) {
    return out << "_" << data.convertOutputPrecision << "_" << data.subtractValues << "_" << data.multiplyValues;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
