// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "subgraph_group_normalization.hpp"

/* The main purpose is to test that GNDecomposition properly decomposes groupNormalization operation
 */

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        PartialShape,                    // Input 0 Shape
        size_t,                          // numGroup
        float                            // epsilon
> GroupNormalizationParams;

class GNDecompositionTest : public LoweringTests, public testing::WithParamInterface<GroupNormalizationParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GroupNormalizationParams> obj);
protected:
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_model;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
