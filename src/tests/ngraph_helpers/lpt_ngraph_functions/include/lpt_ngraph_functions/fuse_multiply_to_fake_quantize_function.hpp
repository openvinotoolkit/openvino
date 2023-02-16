// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"
#include "common/add.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FuseMultiplyToFakeQuantizeFunction {
public:
    static std::shared_ptr<ngraph::Function> get(
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData,
        const DequantizationOperations& dequantization);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
