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
    // Parameter => [service ops for workaround to add FakeQuantize to Snippet] => FakeQuantize => Result
    static std::shared_ptr<ov::Model> get(
        const ngraph::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ngraph::Shape>& fakeQuantizeShapes,
        const float zeroPoint);

    // Parameter => Subgraph (Parameter => FakeQuantize => Result) => Result
    static std::shared_ptr<ov::Model> getSubgraphWithFakeQuantize(
        const ngraph::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ngraph::Shape>& fakeQuantizeShapes,
        const float zeroPoint);

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

