// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class FakeQuantizePrecisionSelectionFunction {
public:
    class ActualValues {
    public:
        bool operationBeforeLimitedOperationIsPrecisionTransparent;
        builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    class ExpectedValues {
    public:
        bool operationBeforeLimitedOperationIsPrecisionTransparent;
        ov::element::Type fakeQuantizeOnDataOutPrecision;
        builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const ActualValues& values);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ExpectedValues& values);
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizePrecisionSelectionFunction::ActualValues& values) {
    return out << values.fakeQuantizeOnData << "_" << values.fakeQuantizeOnWeights;
}

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizePrecisionSelectionFunction::ExpectedValues& values) {
    return out << values.fakeQuantizeOnDataOutPrecision << "_" << values.fakeQuantizeOnData << "_" << values.fakeQuantizeOnWeights;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
