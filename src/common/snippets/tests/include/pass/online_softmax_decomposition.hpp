// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "subgraph_softmax.hpp"

/* The main purpose is to test that OnlineSoftmaxDecomposition is properly decomposed.
 */

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        PartialShape  // Input shape
> OnlineSoftmaxDecompositionTestParams;

class OnlineSoftmaxDecompositionTest : public LoweringTests, public testing::WithParamInterface<OnlineSoftmaxDecompositionTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<OnlineSoftmaxDecompositionTestParams> obj);
protected:
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_model;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
