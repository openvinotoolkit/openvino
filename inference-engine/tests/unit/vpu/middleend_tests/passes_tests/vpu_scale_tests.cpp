// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"
#include <vpu/stages/mx_stage.hpp>
#include <vpu/middleend/hw/utility.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "vpu/private_plugin_config.hpp"
#include "common_test_utils/common_utils.hpp"

using namespace vpu;

namespace LayerTestsDefinitions {
typedef std::tuple<
    std::string
> VpuScaleParams;

class VpuScaleTest : public testing::WithParamInterface<VpuScaleParams>,
                     public GraphTransformerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<VpuScaleParams>& obj);

protected:
    void SetUp() override;
    void Compile() {
        m_pipeline.run(m_testModel.getBaseModel());
    }

protected:
    std::string additionalConfig = {};
    TestModel m_testModel;

private:
    void InitModel() {
        m_testModel = CreateTestModel();
        const DataDesc desc{1};

        m_testModel.createInputs({desc});
        m_testModel.createOutputs({desc});

        auto stage = m_testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork()});
        ASSERT_NO_THROW(Compile());
    }

    void InitPipeline() {
        m_pipeline = PassSet();
        m_pipeline.addPass(passManager->analyzeWeightableLayers());
    }
    PassSet m_pipeline;
};

std::string VpuScaleTest::getTestCaseName(const testing::TestParamInfo<VpuScaleParams>& obj) {
    std::string additionalConfig;
    std::tie(additionalConfig) = obj.param;
    std::ostringstream result;
    result << "_VPUScalePattern=" << additionalConfig;
    return result.str();
}

void VpuScaleTest::SetUp() {
    ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());

    std::tie(additionalConfig) = this->GetParam();
    config.set(InferenceEngine::MYRIAD_SCALES_PATTERN, additionalConfig);

    ASSERT_NO_FATAL_FAILURE(InitCompileEnv());
    ASSERT_NO_FATAL_FAILURE(InitPipeline());
    ASSERT_NO_FATAL_FAILURE(InitModel());
}

TEST_P(VpuScaleTest, IsScaleWorkCorrectly) {
};

// Test cases
INSTANTIATE_TEST_SUITE_P(VPU_Unit_ScaleTest, VpuScaleTest,
                        ::testing::Combine(
                            ::testing::Values(std::string("any:0.2"))),
                        VpuScaleTest::getTestCaseName);
} // namespace LayerTestsDefinitions
