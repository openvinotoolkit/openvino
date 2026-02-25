// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"

namespace ov {
namespace test {
namespace snippets {

class FakeQuantizeFunction {
public:
    // Parameter => Operation => FakeQuantize => Result
    static std::shared_ptr<ov::Model> getOperationAndFakeQuantize(
        const ov::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ov::Shape>& fakeQuantizeShapes,
        const float zeroPoint,
        const std::vector<std::shared_ptr<ov::Node>>& prerequisites,
        std::shared_ptr<ov::Node> operation = nullptr);

    // Parameter => Subgraph (Parameter => FakeQuantize => Result) => Result
    static std::shared_ptr<ov::Model> getSubgraphWithFakeQuantize(
        const ov::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ov::Shape>& fakeQuantizeShapes,
        const float zeroPoint,
        const std::vector<std::shared_ptr<ov::Node>>& prerequisites = {},
        const std::vector<std::shared_ptr<Node>>& beforeFakeQuantizeOperations = {});

    // Parameter => Subgraph (Parameter => element-wise ops from FakeQuantize decomposition results => Result) => Result
    static std::shared_ptr<ov::Model> getSubgraphWithDecomposedFakeQuantize(
        const ov::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ov::Shape>& fakeQuantizeShapes,
        const float zeroPoint);

    static std::shared_ptr<ov::Node> getDecomposedFakeQuantizeOps(
        const ov::Output<ov::Node>& input,
        const ov::element::Type outType,
        float il, float ih, float scale,
        bool doRounding = false,
        bool doDequantize = false);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
