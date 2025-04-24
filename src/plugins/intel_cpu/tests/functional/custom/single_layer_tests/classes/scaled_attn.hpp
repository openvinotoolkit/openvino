// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<ElementType,                      // netPrecision
                   std::vector<InputShape>,          // shape
                   bool,                             // is_causal
                   bool,                             // has_attn
                   bool,                             // has_scale
                   std::string,                      // targetDevice
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
    bool is_causal;
    bool has_attn;
    bool has_scale;
};

}  // namespace test
}  // namespace ov
