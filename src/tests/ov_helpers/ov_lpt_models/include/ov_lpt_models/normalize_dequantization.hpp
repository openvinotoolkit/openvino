// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class NormalizeDequantizationFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations dequantization);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
