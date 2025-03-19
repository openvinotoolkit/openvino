// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "elementwise.hpp"
#include "ov_lpt_models/common/constant.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class MultiplyBranch {
public:
    MultiplyBranch(const PartialShape& inputShape,
                   const ov::builder::subgraph::Constant& constant,
                   const ov::element::Type& input_precision,
                   const ov::builder::subgraph::DequantizationOperations& dequantization,
                   const ov::builder::subgraph::FakeQuantizeOnData& fake_quantize)
                   : inputShape(inputShape),
                     constant(constant),
                     input_precision(input_precision),
                     dequantization(dequantization),
                     fake_quantize(fake_quantize) {}

    PartialShape inputShape;
    ov::builder::subgraph::Constant constant;
    ov::element::Type input_precision;
    ov::builder::subgraph::DequantizationOperations dequantization;
    ov::builder::subgraph::FakeQuantizeOnData fake_quantize;
};

class MultiplyValues {
public:
    MultiplyValues(const MultiplyBranch& branch1,
                   const MultiplyBranch& branch2,
                   const ov::builder::subgraph::DequantizationOperations& after_dequantization)
                   : branch1(branch1), branch2(branch2), after_dequantization(after_dequantization) {}

    MultiplyBranch branch1;
    MultiplyBranch branch2;
    ov::builder::subgraph::DequantizationOperations after_dequantization;
};

class MultiplyFunction : public ElementwiseFunction {
public:
    static std::shared_ptr<ov::Model> get(const ov::element::Type model_precision, const MultiplyValues& actualValues);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
