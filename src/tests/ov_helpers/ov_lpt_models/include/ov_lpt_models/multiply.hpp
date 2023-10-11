// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

#include "elementwise.hpp"
#include "ov_lpt_models/common/constant.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MultiplyBranch {
public:
    MultiplyBranch(const PartialShape& inputShape,
                   const ngraph::builder::subgraph::Constant& constant,
                   const ngraph::element::Type& input_precision,
                   const ngraph::builder::subgraph::DequantizationOperations& dequantization,
                   const ngraph::builder::subgraph::FakeQuantizeOnData& fake_quantize)
                   : inputShape(inputShape),
                     constant(constant),
                     input_precision(input_precision),
                     dequantization(dequantization),
                     fake_quantize(fake_quantize) {}

    PartialShape inputShape;
    ngraph::builder::subgraph::Constant constant;
    ngraph::element::Type input_precision;
    ngraph::builder::subgraph::DequantizationOperations dequantization;
    ngraph::builder::subgraph::FakeQuantizeOnData fake_quantize;
};

class MultiplyValues {
public:
    MultiplyValues(const MultiplyBranch& branch1,
                   const MultiplyBranch& branch2,
                   const ngraph::builder::subgraph::DequantizationOperations& after_dequantization)
                   : branch1(branch1), branch2(branch2), after_dequantization(after_dequantization) {}

    MultiplyBranch branch1;
    MultiplyBranch branch2;
    ngraph::builder::subgraph::DequantizationOperations after_dequantization;
};

class MultiplyFunction : public ElementwiseFunction {
public:
    static std::shared_ptr<ngraph::Function> get(const element::Type model_precision, const MultiplyValues& actualValues);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
