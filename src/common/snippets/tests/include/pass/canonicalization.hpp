// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

using BlockedShape = ngraph::snippets::op::Subgraph::BlockedShape;
using BlockedShapeVector = ngraph::snippets::op::Subgraph::BlockedShapeVector;

// todo: implement tests with 3 inputs and two outputs (aka SnippetsCanonicalizationParams3Inputs)
// Note that the expected output shape isn't necessary equal to one of the output blocked_shapes.
// For example, consider the following graph: (1, 2, 2, 1, 8) + (1, 2, 1, 1, 8) + (1, 2, 1, 5, 8) => (1, 2, 2, 1, 8) + (1, 2, 1, 5, 8).
typedef std::tuple<
        std::tuple<Shape, BlockedShape>, // Shape & BlockedShape for input 0
        std::tuple<Shape, BlockedShape>, // Shape & BlockedShape for input 0
        BlockedShape, // BlockedShape output shape passed to canonicalize()
        Shape // expected output Shape
> canonicalizationParams;


class CanonicalizationTests : public LoweringTests, public testing::WithParamInterface<canonicalizationParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<canonicalizationParams> obj);

protected:
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_function;
    Shape expected_output_shape;
    BlockedShapeVector input_blocked_shapes;
    BlockedShapeVector output_blocked_shapes;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov