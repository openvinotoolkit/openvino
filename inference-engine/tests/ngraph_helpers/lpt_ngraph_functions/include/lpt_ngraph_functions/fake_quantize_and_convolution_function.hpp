// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FakeQuantizeAndConvolutionFunction {
public:
    // TODO: move to ConvolutionFunction
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData,
        const FakeQuantizeOnWeights& fakeQuantizeOnWeights);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
