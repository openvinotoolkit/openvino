// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>
#include "fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FakeQuantizeOnWeights: public FakeQuantizeOnData {
public:
    FakeQuantizeOnWeights();

    FakeQuantizeOnWeights(
        const uint64_t quantizationLevel,
        const ngraph::Shape& constantShape,
        const std::vector<float>& inputLowValues,
        const std::vector<float>& inputHighValues,
        const std::vector<float>& outputLowValues,
        const std::vector<float>& outputHighValues,
        const ngraph::element::Type outputPrecision = ngraph::element::undefined);

    virtual ~FakeQuantizeOnWeights();

    bool empty() const override;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeOnWeights& data) {
    return out << "_" << data.quantizationLevel << "_" << data.constantShape << "_" << data.outputLowValues << "_" << data.outputHighValues;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
