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

class DepthToSpaceFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(const ngraph::element::Type ngPrecision, const ngraph::Shape& inputShape);
    static std::shared_ptr<ngraph::Function> getReference(const ngraph::element::Type ngPrecision, const ngraph::Shape& inputShape);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
