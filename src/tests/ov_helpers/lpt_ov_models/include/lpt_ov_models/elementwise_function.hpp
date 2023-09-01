// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <low_precision/layer_transformation.hpp>

#include "lpt_ov_models/common/builders.hpp"
#include "lpt_ov_models/common/convolution.hpp"
#include "lpt_ov_models/common/dequantization_operations.hpp"
#include "lpt_ov_models/common/fake_quantize_on_data.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class ElementwiseFunction {
public:
    static std::shared_ptr<ov::Model> getOriginalSubgraphWithConvolutions(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const bool broadcast,
        const std::string& elementWiseType,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataBefore1,
        const ov::builder::subgraph::Convolution& convolution1,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter1,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataBefore2,
        const ov::builder::subgraph::Convolution& convolution2,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter2,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
