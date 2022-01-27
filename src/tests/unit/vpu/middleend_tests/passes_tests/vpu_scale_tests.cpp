// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/hw/utility.hpp>
#include <vpu/stages/mx_stage.hpp>

#include "common_test_utils/common_utils.hpp"
#include "graph_transformer_tests.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "vpu/configuration/options/vpu_scales_option.hpp"
#include "vpu/private_plugin_config.hpp"

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 9) && !defined(__clang__) && !defined(IE_GCC_4_8)
#define IE_GCC_4_8
#endif

using namespace vpu;
IE_SUPPRESS_DEPRECATED_START
namespace LayerTestsDefinitions {
typedef std::tuple<std::string> VpuScaleParams;

class VpuScaleTest : public testing::WithParamInterface<VpuScaleParams>,
                     public GraphTransformerTest {
protected:
    void SetUp() override;
    void Compile() {
        m_pipeline.run(m_testModel);
    }
    Model m_testModel;


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

        auto conv = std::make_shared<ie::ConvolutionLayer>(ie::LayerParams{"conv1", "Convolution", ie::Precision::FP16});
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
        Data output2;

        output2 = m_testModel->addOutputData(
            "Output",
            DataDesc(
                DataType::FP16, DimsOrder::NCHW,
                {(inputX + padx_begin + padx_end - kernelx) / kernelStrideX + 1,
                 (inputY + pady_begin + pady_end - kernely) / kernelStrideY + 1,
                 8, 1}));
        auto conv2 = std::make_shared<ie::ConvolutionLayer>(ie::LayerParams{"conv2", "Convolution", ie::Precision::FP16});
        conv2->_kernel_x = kernelx;
        conv2->_kernel_y = kernely;
        conv2->_stride_x = kernelStrideX;
        conv2->_stride_y = kernelStrideY;
        conv2->_dilation_x = dilationX;
        conv2->_dilation_x = dilationY;

        conv2->_padding.insert(0, padx_begin);
        conv2->_padding.insert(1, pady_begin);
        conv2->_pads_end.insert(0, padx_end);
        conv2->_pads_end.insert(1, pady_end);
        conv2->_auto_pad = "same_upper";

        conv2->_weights = ie::make_shared_blob<short>({ ie::Precision::FP16, {static_cast<size_t>(kernelx * kernely * 8 * 8)}, ie::Layout::C });
        conv2->_weights->allocate();

        frontEnd->parseConvolution(m_testModel, conv2, {output}, {output2});
    }

    void InitPipeline() {
        m_pipeline = PassSet();
        m_pipeline.addPass(passManager->analyzeWeightableLayers());
    }
    PassSet m_pipeline;
};

void VpuScaleTest::SetUp() {
}

TEST_F(VpuScaleTest, IsScaleWorkCorrectly) {
#ifdef IE_GCC_4_8
    GTEST_SKIP();
#endif
    std::string configValue = "conv1:0.2; conv2:1.4";

    ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());
    config.set(InferenceEngine::MYRIAD_SCALES_PATTERN, configValue);
    ASSERT_NO_FATAL_FAILURE(InitCompileEnv());
    ASSERT_NO_FATAL_FAILURE(InitPipeline());
    ASSERT_NO_FATAL_FAILURE(InitModel());
    ASSERT_NO_THROW(Compile());

    for (const auto& stage : m_testModel->getStages()) {
        auto scale = stage->attrs().getOrDefault<float>("scaleFactor");
        if (stage->name() == "conv1") {
            ASSERT_FLOAT_EQ(scale, 0.2);
            continue;
        }
        if (stage->name() == "conv2") {
            ASSERT_FLOAT_EQ(scale, 1.4);
        }
    }
}

TEST_F(VpuScaleTest, IsRegexInScaleWorksCorrectly) {
#ifdef IE_GCC_4_8
    GTEST_SKIP();
#endif
    std::string configValue = "conv1:0.2";
    ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());
    config.set(InferenceEngine::MYRIAD_SCALES_PATTERN, configValue);

    ASSERT_NO_FATAL_FAILURE(InitCompileEnv());
    ASSERT_NO_FATAL_FAILURE(InitPipeline());
    ASSERT_NO_FATAL_FAILURE(InitModel());
    ASSERT_NO_THROW(Compile());

    for (const auto& stage : m_testModel->getStages()) {
        auto scale = stage->attrs().getOrDefault<float>("scaleFactor");
        if (stage->name() == "conv1") {
            ASSERT_FLOAT_EQ(scale, 0.2);
            continue;
        }
    }
}
}  // namespace LayerTestsDefinitions
