// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"


namespace ngraph {
namespace builder {
namespace subgraph {

class FakeQuantizeFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fakeQuantizeOnData,
        const bool addNotPrecisionPreservedOperation);

    static std::shared_ptr<ngraph::Function> getOriginalWithMaxPool(
            const ngraph::element::Type precision,
            const ngraph::PartialShape& inputShape,
            const FakeQuantizeOnData& fakeQuantizeOnData);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const bool updatePrecisions,
        const FakeQuantizeOnDataWithConstant& fakeQuantizeOnData,
        const ngraph::element::Type fakeQuantizeOutputPrecision,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const bool addNotPrecisionPreservedOperation);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
