// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "subgraph_softmax.hpp"

/* The main purpose is to test that SoftmaxDecomposition properly decomposes Softmax operation
 */

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        PartialShape,  // Input shape
        int,          // Softmax axis
        SoftmaxVersion
> SoftmaxDecompositionTestParams;

class SoftmaxDecompositionTest : public LoweringTests, public testing::WithParamInterface<SoftmaxDecompositionTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SoftmaxDecompositionTestParams> obj);
protected:
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_model;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
