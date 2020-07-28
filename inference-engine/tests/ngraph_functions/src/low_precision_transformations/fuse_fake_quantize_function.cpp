// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fuse_fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::get(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization,
    const ngraph::element::Type precisionFqOnData,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const std::shared_ptr<Node> lastDequantization = makeDequantization(input, dequantization);
    const std::shared_ptr<Node> fakeQuantize = precisionAfterDequantization == precisionFqOnData ?
            makeFakeQuantize(lastDequantization, precisionFqOnData, fqOnData) :
            makeFakeQuantizeTypeRelaxed(lastDequantization, precisionFqOnData, fqOnData);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FuseFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
