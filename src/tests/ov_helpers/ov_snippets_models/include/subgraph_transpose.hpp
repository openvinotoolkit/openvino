// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "snippets_helpers.hpp"

/* This file contains definitions of relatively simple functions (models) that will be used
 * to test snippets-specific behavior. All the functions are expected to be direct descendants of
 * SnippetsFunctionBase, so their constructors take only one (inputShapes) argument.
 */

namespace ov {
namespace test {
namespace snippets {
/// Minimal graph to test Transpose support: Parameter->Transpose->Result
/// Tokenized simply by starting subgraph, supported through TransposeDecomposition
//   in1        Const(order)
//        Transpose
//         Result
class TransposeFunction : public SnippetsFunctionBase {
public:
    explicit TransposeFunction(const std::vector<PartialShape>& inputShapes, std::vector<int> order)
    : SnippetsFunctionBase(inputShapes), order(std::move(order)) {
        NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
    std::vector<int> order;
};
/// Testing Transpose + Eltwise support on the example of Mul op
/// Tokenized simply by starting subgraph, supported through TransposeDecomposition
//   in1        Const(order)
//        Transpose
//   in2     |
//        Multiply
//         Result
class TransposeMulFunction : public SnippetsFunctionBase {
public:
    explicit TransposeMulFunction(const std::vector<PartialShape>& inputShapes, std::vector<int> order)
            : SnippetsFunctionBase(inputShapes), order(std::move(order)) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::vector<int> order;
};
}  // namespace snippets
}  // namespace test
}  // namespace ov
