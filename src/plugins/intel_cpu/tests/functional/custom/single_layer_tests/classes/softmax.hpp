// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

struct SoftMaxConfig {
    ov::test::InputShape inputShape;
    size_t axis;
};

typedef std::tuple<ElementType,    // netPrecision
                   SoftMaxConfig,  // softmaxTestConfig
                   std::string,    // targetDevice
                   CPUSpecificParams,
                   ov::AnyMap> //device_config
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
}  // namespace test
}  // namespace ov
