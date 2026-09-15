// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

// Graph that exercises the parameter deduplication path in tokenization.
// Two eltwise Subgraphs share the same Split output.  When the downstream
// Add (Add2) merges them, the shared external_input must be reused instead
// of creating a duplicate body parameter.
//
//   param0 -> Split(axis=0, num_splits=2)
//     split->output(0) + const1 -> Add
//                                              Add2 -> Result
//     split->output(0) * const2 -> Mul
//
// After tokenization, Add, Mul and Add2 are collapsed into a single
// Subgraph.  The shared input split->output(0) appears exactly once in
// external_inputs, and the body parameter for the Mul branch is replaced
// with the one already created for the Add branch.
class CommonParentTokenizationFunction : public SnippetsFunctionBase {
public:
    explicit CommonParentTokenizationFunction(const std::vector<PartialShape>& inputShapes)
        : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
