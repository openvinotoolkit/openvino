// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        ov::PartialShape,          // Input 0 Shape
        ov::PartialShape,          // Input 1 Shape
        ov::PartialShape,          // Input 2 Shape
        size_t,                    // Add input index
        bool,                      // Constant input
        size_t,                    // Expected num nodes
        size_t,                    // Expected num subgraphs
        std::string                // Target Device
> FusedMulAddParams;

class FusedMulAdd : public testing::WithParamInterface<ov::test::snippets::FusedMulAddParams>,
                   virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::FusedMulAddParams> obj);

protected:
    void SetUp() override;
};


} // namespace snippets
} // namespace test
} // namespace ov
