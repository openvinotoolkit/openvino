// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>

#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

class NormalizeL2Function {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const std::pair<ngraph::Shape, ngraph::Shape>& shapes,
        const ngraph::element::Type precisionOnActivation,
        const bool fuseMultiply,
        const bool shift);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const std::pair<ngraph::Shape, ngraph::Shape>& shapes,
        const ngraph::element::Type precisionOnActivation,
        const bool fuseMultiply,
        const bool shift);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
