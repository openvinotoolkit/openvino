// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_models/builders.hpp>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

struct SoftMaxConfig {
    ov::test::InputShape inputShape;
    size_t axis;
};

typedef std::tuple<ElementType,    // netPrecision
                   SoftMaxConfig,  // softmaxTestConfig
                   std::string,    // targetDevice
                   CPUSpecificParams>
    softmaxCPUTestParams;

class SoftMaxLayerCPUTest : public testing::WithParamInterface<softmaxCPUTestParams>,
                            virtual public SubgraphBaseTest,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<softmaxCPUTestParams>& obj);

protected:
    void SetUp() override;
};

namespace SoftMax {


}  // namespace SoftMax
}  // namespace CPULayerTestsDefinitions
