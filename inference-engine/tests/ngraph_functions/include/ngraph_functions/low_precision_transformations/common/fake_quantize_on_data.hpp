// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/layer_transformation.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FakeQuantizeOnData {
public:
    FakeQuantizeOnData();

    FakeQuantizeOnData(
        const size_t quantizationLevel,
        const ngraph::Shape& constantShape,
        const std::vector<float>& lowValues,
        const std::vector<float>& highValues);

    virtual ~FakeQuantizeOnData();

    bool isSigned() const;
    virtual bool empty() const;

    size_t quantizationLevel;
    ngraph::Shape constantShape;
    std::vector<float> lowValues;
    std::vector<float> highValues;
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

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeOnData& data) {
    return out << "_" << data.constantShape << "_" << data.lowValues << "_" << data.highValues;
}

}
}
}
