// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include "low_precision/layer_transformation.hpp"
#include "common/add.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class FuseSubtractToFakeQuantizeFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData,
        const DequantizationOperations& dequantization);

    static std::shared_ptr<ov::Model> get(
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData,
        const DequantizationOperations& dequantization,
        const FakeQuantizeOnDataWithConstant& fqOnData2,
        const DequantizationOperations& dequantization2);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
