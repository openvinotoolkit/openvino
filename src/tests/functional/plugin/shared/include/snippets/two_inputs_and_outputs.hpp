// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<InputShape>,     // Input Shape All shapes
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> TwoInputsAndOutputsParams;

class TwoInputsAndOutputs : public testing::WithParamInterface<ov::test::snippets::TwoInputsAndOutputsParams>,
                            virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::TwoInputsAndOutputsParams> obj);

protected:
    void SetUp() override;
};

// TwoInputsAndOutputsWithReversedOutput tests the same network with reversed order of Result nodes.
// It changes order of nodes after topological sort. The test checks the correctness of the
// algorithm for checking possible cyclic dependency for nodes with Result node in consumers in tokenization.
class TwoInputsAndOutputsWithReversedOutputs : public TwoInputsAndOutputs {
protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov
