// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<ov::PartialShape>, // Input shapes
        size_t,                        // Expected num nodes
        size_t,                        // Expected num subgraphs
        std::string                    // Target Device
> EnforcePrecisionTestParams;

class EnforcePrecisionTest :
    public testing::WithParamInterface<EnforcePrecisionTestParams>,
    virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EnforcePrecisionTestParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov
