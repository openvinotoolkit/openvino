// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/builders.hpp"


namespace ov {
namespace builder {
namespace subgraph {

class FakeQuantizeFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::pass::low_precision::LayerTransformation::Params& params,
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fakeQuantizeOnData,
        const bool addNotPrecisionPreservedOperation);

    static std::shared_ptr<ov::Model> getOriginalWithMaxPool(
            const ov::element::Type precision,
            const ov::PartialShape& inputShape,
            const FakeQuantizeOnData& fakeQuantizeOnData);

    static std::shared_ptr<ov::Model> getReference(
        const ov::pass::low_precision::LayerTransformation::Params& params,
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const bool updatePrecisions,
        const FakeQuantizeOnDataWithConstant& fakeQuantizeOnData,
        const ov::element::Type fakeQuantizeOutputPrecision,
        const ov::builder::subgraph::DequantizationOperations& dequantization,
        const bool addNotPrecisionPreservedOperation);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
