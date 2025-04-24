// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        InputShape,                      // Input 0 Shape
        size_t,                          // numGroup
        float,                           // epsilon
        size_t,                          // Expected num nodes
        size_t,                          // Expected num subgraphs
        std::string                      // Target Device
> GroupNormalizationParams;

class GroupNormalization : public testing::WithParamInterface<ov::test::snippets::GroupNormalizationParams>,
                           virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::GroupNormalizationParams> obj);

protected:
    void SetUp() override;
    InputShape ExtractScaleShiftShape(const InputShape& shape);
};

} // namespace snippets
} // namespace test
} // namespace ov