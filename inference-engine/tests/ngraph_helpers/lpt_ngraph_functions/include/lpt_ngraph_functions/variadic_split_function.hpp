// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/ngraph.hpp>
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"



namespace ngraph {
namespace builder {
namespace subgraph {

class VariadicSplitFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const int64_t splitedAxis,
        const std::vector<size_t>& splitLengths);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
        const int64_t splitedAxis,
        const std::vector<size_t>& splitLengths);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type precisionAfterOperation,
        const std::vector<ngraph::builder::subgraph::DequantizationOperations>& dequantizationAfter,
        const int64_t splitedAxis,
        const std::vector<size_t>& splitLengths);
};
inline std::ostream& operator<<(std::ostream& os,
    const std::vector<size_t>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
