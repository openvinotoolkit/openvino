// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "ov_models/builders.hpp"

using namespace ov::test;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using ConvPoolActivTestParams = fusingSpecificParams;

class ConvPoolActivTest : public testing::WithParamInterface<ConvPoolActivTestParams>, public CpuTestWithFusing,
                          virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvPoolActivTestParams> obj);
protected:
    void SetUp() override;
};
} // namespace SubgraphTestsDefinitions
