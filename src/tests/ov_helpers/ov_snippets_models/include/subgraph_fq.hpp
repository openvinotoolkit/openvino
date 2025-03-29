// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "./snippets_helpers.hpp"

/* This file contains definitions of relatively simple functions (models) that will be used
 * to test snippets-specific behavior. All the functions are expected to be direct descendants of
 * SnippetsFunctionBase, so their constructors take only one (inputShapes) argument.
 */

namespace ov {
namespace test {
namespace snippets {
/// Non-supported potential count of Constants after FQ decomposition
//    in1
//    FQ - 3 Constants ---- Subgraph0 with 11 IO
//    FQ - 6 Constants --/
//    FQ - 3 Constants ---- Subgraph1 with 4 IO
//   Result
class ThreeFQFunction : public SnippetsFunctionBase {
public:
    explicit ThreeFQFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase({{1, 3, 20, 20}}) {}
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};
}  // namespace snippets
}  // namespace test
}  // namespace ov
