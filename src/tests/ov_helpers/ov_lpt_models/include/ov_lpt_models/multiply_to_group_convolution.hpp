// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "ov_lpt_models/common/constant.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MultiplyToGroupConvolutionFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type& precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const bool haveMultiplyWithNoConstBeforeDequantization);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData,
        const Constant& constant,
        const bool parentHasOneConsumer = true);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type& precision,
        const std::shared_ptr<ngraph::opset1::Constant>& weights,
        const std::shared_ptr<ngraph::opset1::Constant>& biases,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
