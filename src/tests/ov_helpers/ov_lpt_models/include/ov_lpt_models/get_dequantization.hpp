// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <vector>
#include <ngraph/ngraph.hpp>
#include <low_precision/common/fake_quantize_dequantization.hpp>
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class GetDequantizationFunction {
public:
    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type& precision,
        const Shape& shape,
        const FakeQuantizeOnData& fakeQuantize,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore);

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type& precision,
        const Shape& shape,
        const FakeQuantizeOnData& fakeQuantize,
        const ov::pass::low_precision::FakeQuantizeDequantization& dequantization);

    static std::shared_ptr<ngraph::Function> getOriginal(
        bool isConvert, bool isSubtract, size_t subDataInput, size_t mulDataInput);

    static std::shared_ptr<ngraph::Function> getReference(
        ov::pass::low_precision::FakeQuantizeDequantization dequantization);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
