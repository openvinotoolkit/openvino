// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<InputShape>,         // Input shapes
        std::vector<float>,              // FakeQuantize intervals
        size_t,                          // Expected num nodes
        size_t,                          // Expected num subgraphs
        std::string                      // Target Device
> PrecisionPropagationParams;

class PrecisionPropagationConvertion :
    public testing::WithParamInterface<PrecisionPropagationParams>,
    virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PrecisionPropagationParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov
