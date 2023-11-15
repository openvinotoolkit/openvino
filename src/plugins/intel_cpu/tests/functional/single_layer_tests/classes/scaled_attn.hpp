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

struct ScaledAttnConfig {
    ov::test::InputShape inputShape;
    bool is_causal;                    // causal
    bool has_attn;                     // if attention_mask valid
    bool has_scale;                    // if scale valid
};

typedef std::tuple<ElementType,         // netPrecision
                   ScaledAttnConfig,    // ScaledAttnTestConfig
                   std::string,         // targetDevice
                   CPUSpecificParams>
    ScaledAttnCPUTestParams;

class ScaledAttnLayerCPUTest : public testing::WithParamInterface<ScaledAttnCPUTestParams>,
                            virtual public SubgraphBaseTest,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ScaledAttnCPUTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    ScaledAttnConfig config;
};

namespace ScaledAttn {


}  // namespace ScaledAttn
}  // namespace CPULayerTestsDefinitions
