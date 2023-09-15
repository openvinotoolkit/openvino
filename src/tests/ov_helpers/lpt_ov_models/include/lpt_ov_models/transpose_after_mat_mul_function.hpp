// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

#include "lpt_ov_models/common/dequantization_operations.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "lpt_ov_models/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class TransposeAfterMatMulFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
