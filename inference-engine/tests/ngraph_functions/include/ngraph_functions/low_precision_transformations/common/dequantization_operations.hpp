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
    class Convert {
    public:
        Convert();
        Convert(const ngraph::element::Type outPrecision);
        bool empty() const noexcept;

        ngraph::element::Type outPrecision;
    private:
        bool isEmpty;
    };

    DequantizationOperations();

    DequantizationOperations(
        const Convert& convert,
        const std::vector<float>& subtractValues,
        const std::vector<float>& multiplyValues);

    bool empty() const;

    Convert convert;
    std::vector<float> subtractValues;
    std::vector<float> multiplyValues;
};

inline std::ostream& operator<<(std::ostream& out, const DequantizationOperations& data) {
    return out << "_" <<
        data.convert.outPrecision << "_" <<
        data.subtractValues << "_" <<
        data.multiplyValues;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
