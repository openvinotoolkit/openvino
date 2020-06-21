// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

class MatMulFunction {
public:
    // TODO: move to base class
    static std::vector<std::shared_ptr<ngraph::op::Parameter>> getInputs(const std::vector<std::shared_ptr<ngraph::Node>>& nodes);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type ngPrecision,
        const ngraph::Shape& inputShape,
        const std::vector<std::shared_ptr<ngraph::Node>>& nodes);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type ngPrecision,
        const ngraph::Shape& inputShape,
        const std::vector<std::shared_ptr<ngraph::Node>>& nodes);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
