// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<Node> makeDequantization(
    const std::shared_ptr<ngraph::Node> data,
    const DequantizationOperations& dequantizationOperations);

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantize(
    const std::shared_ptr<ngraph::Node>& input,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData);

// TODO: refactor
std::shared_ptr<Node> makeFakeQuantizeTypeRelaxed(
    const std::shared_ptr<ngraph::Node>& input,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData);

} // namespace subgraph
} // namespace builder
} // namespace ngraph
