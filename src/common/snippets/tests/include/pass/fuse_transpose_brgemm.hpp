// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets_helpers.hpp"

/* The main purpose is to test that FuseTransposeBrgemm properly fuses 0213 Transposes on both inputs, as well as on output
 */

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<PartialShape>, // Input shapes
        size_t                     // Transpose position
> fuseTransposeBrgemmParams;

class FuseTransposeBrgemmTests : public LoweringTests, public testing::WithParamInterface<fuseTransposeBrgemmParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<fuseTransposeBrgemmParams> obj);
protected:
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_model;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
