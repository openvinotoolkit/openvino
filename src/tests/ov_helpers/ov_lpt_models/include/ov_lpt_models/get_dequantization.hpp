// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <vector>
#include <low_precision/common/fake_quantize_dequantization.hpp>
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class GetDequantizationFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const ov::element::Type& precision,
        const Shape& shape,
        const FakeQuantizeOnData& fakeQuantize,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore);

    static std::shared_ptr<ov::Model> get(
        const ov::element::Type& precision,
        const Shape& shape,
        const FakeQuantizeOnData& fakeQuantize,
        const ov::pass::low_precision::FakeQuantizeDequantization& dequantization);

    static std::shared_ptr<ov::Model> getOriginal(
        bool isConvert, bool isSubtract, size_t subDataInput, size_t mulDataInput);

    static std::shared_ptr<ov::Model> getReference(
        ov::pass::low_precision::FakeQuantizeDequantization dequantization);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ov
