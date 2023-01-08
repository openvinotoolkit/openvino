// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <low_precision/layer_transformation.hpp>

#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/convolution.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ElementwiseFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginalSubgraphWithConvolutions(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const bool broadcast,
        const std::string& elementWiseType,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataBefore1,
        const ngraph::builder::subgraph::Convolution& convolution1,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter1,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataBefore2,
        const ngraph::builder::subgraph::Convolution& convolution2,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter2,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
