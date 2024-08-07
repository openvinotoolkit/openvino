// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>

#include "snippets/pass/tokenization.hpp"
#include "subgraph_group_normalization.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        PartialShape,                    // Input 0 Shape
        size_t,                          // numGroup
        float                            // epsilon
> GroupNormalizationParams;

class TokenizeGNSnippetsTests : public TransformationTestsF, public testing::WithParamInterface<GroupNormalizationParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GroupNormalizationParams> obj);
protected:
    void SetUp() override;
    std::shared_ptr<GroupNormalizationFunction> snippets_model;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
