// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"

namespace ov {
namespace test {
namespace snippets {

class FakeQuantizeFunction {
public:
    // Parameter => Operation => FakeQuantize => Result
    static std::shared_ptr<ov::Model> getOperationAndFakeQuantize(
        const ngraph::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ngraph::Shape>& fakeQuantizeShapes,
        const float zeroPoint,
        const std::vector<std::shared_ptr<ngraph::Node>>& prerequisites,
        std::shared_ptr<ngraph::Node> operation = nullptr);

    // Parameter => Subgraph (Parameter => FakeQuantize => Result) => Result
    static std::shared_ptr<ov::Model> getSubgraphWithFakeQuantize(
        const ngraph::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ngraph::Shape>& fakeQuantizeShapes,
        const float zeroPoint,
        const std::vector<std::shared_ptr<ngraph::Node>>& prerequisites = {},
        const std::vector<std::shared_ptr<Node>>& beforeFakeQuantizeOperations = {});

    // Parameter => Subgraph (Parameter => element-wise ops from FakeQuantize decomposition results => Result) => Result
    static std::shared_ptr<ov::Model> getSubgraphWithDecomposedFakeQuantize(
        const ngraph::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ngraph::Shape>& fakeQuantizeShapes,
        const float zeroPoint);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
