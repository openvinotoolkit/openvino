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
IE_SUPPRESS_DEPRECATED_START
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
        m_pipeline.run(m_testModel);
    }

protected:
    std::string additionalConfig = {};
    Model m_testModel;

private:
    void InitModel() {
        int kernelx = 16;
        int kernely = 1;
        int kernelStrideX = 1;
        int kernelStrideY = 1;
        int dilationX = 1;
        int dilationY = 1;
        int padx_begin = 7;
        int pady_begin = 0;
        int padx_end = 8;
        int pady_end = 0;
        m_testModel = CreateModel();
        int inputX = 32;
        int inputY = 32;

        auto input = m_testModel->addInputData(
            "Input",
            DataDesc(DataType::FP16, DimsOrder::NCHW, {inputX, inputY, 8, 1}));
        m_testModel->attrs().set<int>("numInputs", 1);

        Data output;

        output = m_testModel->addOutputData(
            "Output",
            DataDesc(
                DataType::FP16, DimsOrder::NCHW,
                {(inputX + padx_begin + padx_end - kernelx) / kernelStrideX + 1,
                 (inputY + pady_begin + pady_end - kernely) / kernelStrideY + 1,
                 8, 1}));

        auto conv = std::make_shared<ie::ConvolutionLayer>(ie::LayerParams{"conv", "Convolution", ie::Precision::FP16});
        conv->_kernel_x = kernelx;
        conv->_kernel_y = kernely;
        conv->_stride_x = kernelStrideX;
        conv->_stride_y = kernelStrideY;
        conv->_dilation_x = dilationX;
        conv->_dilation_x = dilationY;

        conv->_padding.insert(0, padx_begin);
        conv->_padding.insert(1, pady_begin);
        conv->_pads_end.insert(0, padx_end);
        conv->_pads_end.insert(1, pady_end);
        conv->_auto_pad = "same_upper";

        conv->_weights = ie::make_shared_blob<short>({ ie::Precision::FP16, {static_cast<size_t>(kernelx * kernely * 8 * 8)}, ie::Layout::C });
        conv->_weights->allocate();

        frontEnd->parseConvolution(m_testModel, conv, {input}, {output});
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
