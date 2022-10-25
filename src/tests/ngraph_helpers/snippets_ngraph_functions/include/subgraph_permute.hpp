// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "./snippets_helpers.hpp"

/* This file contains definitions of relatively simple functions (models) that will be used
 * to test snippets-specific behavior. All the functions are expected to be direct descendants of
 * SnippetsFunctionBase, so their constructors take only one (inputShapes) argument.
 */

namespace ov {
namespace test {
namespace snippets {
/// Add separated from inputs by Sinh to WA CPU-specific disabling after inputs.
/// Works because Sinh is not supported by tokenization yet.
/// Tokenized simply by starting subgraph.
//   in1       in2
//   Sinh       Sinh
//        Add
//      Result
// todo: remove Sinh once "no subgraph after input" limitation is relaxed
class TransposeSinhFunction : public SnippetsFunctionBase {
public:
    explicit TransposeSinhFunction(const std::vector<PartialShape>& inputShapes, std::vector<int> order)
    : SnippetsFunctionBase(inputShapes), order(std::move(order)) {
        NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
    std::vector<int> order;
};
}  // namespace snippets
}  // namespace test
}  // namespace ov
