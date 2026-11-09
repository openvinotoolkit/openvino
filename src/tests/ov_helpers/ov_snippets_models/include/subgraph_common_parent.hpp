// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

// Graph that exercises parameter deduplication at external input indices 0
// and 1 during tokenization.
//
//   param0 -> VariadicSplit(axis=0, lengths=[1, 2])
//     split->output(1) + split->output(0) -> Add
//                                                Add2 -> Result
//     split->output(0) * split->output(1) -> Mul
//
// The Add branch inserts external inputs in [1, 0] order. The Multiply branch
// visits them in [0, 1] order, so tokenization must reuse parameters by both
// external input indices.
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
