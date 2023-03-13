// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        InputShape,                  // Input 0 Shape
        InputShape,                  // Input 1 Shape
        InputShape,                  // Input 2 Shape
        InputShape,                  // Input 3 Shape
        InputShape,                  // Input 4 Shape
        InputShape,                  // Input 5 Shape
        InputShape,                  // Input 6 Shape
        InputShape,                  // Input 7 Shape
        ov::element::Type,           // Element type
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> EdgeReplaceParams;

class EdgeReplace : public testing::WithParamInterface<ov::test::snippets::EdgeReplaceParams>,
                    virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::EdgeReplaceParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov