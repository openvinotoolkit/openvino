// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/convolution.hpp>
#include "../test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph;
using namespace details;

namespace CPULayerTestsDefinitions {

using convLayerCPUTestParamsSet = std::tuple<LayerTestsDefinitions::convLayerTestParamsSet,
                                             CPUSpecificParams,
                                             size_t,   // input offset padding
                                             size_t>;  // output offset padding

class ConvolutionLayerCPUTest : public testing::WithParamInterface<convLayerCPUTestParamsSet>,
                                virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    size_t inOffsetPadding, outOffsetPadding;

    static std::string getTestCaseName(testing::TestParamInfo<convLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::convLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        size_t inOffsetPadding, outOffsetPadding;
        std::tie(basicParamsSet, cpuParams, inOffsetPadding, outOffsetPadding) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::ConvolutionLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::convLayerTestParamsSet>(
                                                                                                                                  basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << "_inputOffsetPadding=" << inOffsetPadding;
        result << "_outputOffsetPadding=" << outOffsetPadding;

        return result.str();
    }

    void ConfigureNetwork() override {
        const auto &inpInfo = cnnNetwork.getInputsInfo();
        auto in = inpInfo.begin();
        TensorDesc descIn = createTensorDesc(in->second->getTensorDesc().getDims(), in->second->getPrecision(), inFmts[0], inOffsetPadding);
        in->second->setInputData(std::make_shared<Data>(in->second->name(), descIn));

        const auto &outInfo = cnnNetwork.getOutputsInfo();
        auto out = outInfo.begin();
        TensorDesc descOut = createTensorDesc(out->second->getTensorDesc().getDims(), out->second->getPrecision(), outFmts[0], outOffsetPadding);
        *(out->second) = *std::make_shared<Data>(out->second->getName(), descOut);
    }

    void Infer() override {
        inferRequest = executableNetwork.CreateInferRequest();
        inputs.clear();

        const auto inputInfo = executableNetwork.GetInputsInfo().begin()->second;
        size_t sizeIn = std::accumulate(inputInfo->getTensorDesc().getBlockingDesc().getBlockDims().begin(),
                                        inputInfo->getTensorDesc().getBlockingDesc().getBlockDims().end(), (size_t)1, std::multiplies<size_t>());
        sizeIn += inputInfo->getTensorDesc().getBlockingDesc().getOffsetPadding();
        preAllocMemIn.resize(sizeIn);
        std::default_random_engine random(1);
        std::uniform_int_distribution<int32_t> distribution(1, 10);
        for (size_t i = 0; i < preAllocMemIn.size(); i++)
            preAllocMemIn[i] = static_cast<float>(distribution(random));
        auto blobIn = make_shared_blob<float>(inputInfo->getTensorDesc(), preAllocMemIn.data(), sizeIn);
        inferRequest.SetBlob(inputInfo->name(), blobIn);

        Layout layout;
        if (inputInfo->getTensorDesc().getDims().size() == 4)
            layout = Layout::NCHW;
        else if (inputInfo->getTensorDesc().getDims().size() == 5)
            layout = Layout::NCDHW;
        else
            throw std::runtime_error("CPU tests: convolution support only 4D and 5D input dims");
        auto refBlob = make_shared_blob<float>(TensorDesc(Precision::FP32, inputInfo->getTensorDesc().getDims(), layout));
        refBlob->allocate();
        const float *src = blobIn->cbuffer().as<const float *>();
        float *dst = refBlob->buffer().as<float *>();
        for (size_t i = 0; i < refBlob->size(); i++) {
            dst[i] = src[blobIn->getTensorDesc().offset(i)];
        }
        inputs.push_back(refBlob);

        const auto outputInfo = executableNetwork.GetOutputsInfo().begin()->second;
        size_t sizeOut = std::accumulate(outputInfo->getTensorDesc().getBlockingDesc().getBlockDims().begin(),
                                         outputInfo->getTensorDesc().getBlockingDesc().getBlockDims().end(), (size_t)1, std::multiplies<size_t>());
        sizeOut += outputInfo->getTensorDesc().getBlockingDesc().getOffsetPadding();
        preAllocMemOut.resize(sizeOut);
        auto blobOut = make_shared_blob<float>(outputInfo->getTensorDesc(), preAllocMemOut.data(), sizeOut);
        inferRequest.SetBlob(outputInfo->getName(), blobOut);

        inferRequest.Infer();
    }

    void Compare(const std::vector<std::uint8_t> &expected, const Blob::Ptr &actual) override {
        auto blobToCmp = make_shared_blob<float>(TensorDesc(actual->getTensorDesc().getPrecision(), actual->getTensorDesc().getDims(),
                                                 TensorDesc::getLayoutByDims(actual->getTensorDesc().getDims())));
        blobToCmp->allocate();
        const float *actualBuffer = actual->cbuffer().as<const float *>();
        float *cmpBuffer = blobToCmp->buffer().as<float *>();
        const auto size = blobToCmp->size();
        for (size_t i = 0; i < size; i++)
            cmpBuffer[i] = actualBuffer[actual->getTensorDesc().offset(i)];

        LayerTestsCommon::Compare(reinterpret_cast<const float *>(expected.data()), cmpBuffer, size, threshold);
    }

protected:
    void SetUp() override {
        LayerTestsDefinitions::convLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams, inOffsetPadding, outOffsetPadding) = this->GetParam();
        LayerTestsDefinitions::convSpecificParams convParams;
        SizeVector dims;
        Precision inPrec;
        std::tie(convParams, inPrec, inPrc, outPrc, inLayout, outLayout, dims, targetDevice) = basicParamsSet;
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChanel;
        op::PadType padType;
        std::tie(kernel, stride, padBegin, padEnd, dilation, numOutChanel, padType) = convParams;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
        auto params = builder::makeParams(ngPrc, {dims});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));
        auto conv = std::dynamic_pointer_cast<opset5::Convolution>(builder::makeConvolution(paramOuts[0], ngPrc, kernel, stride, padBegin, padEnd,
                                                                                            dilation, padType, numOutChanel));
        conv->get_rt_info() = getCPUInfo();
        ResultVector results{std::make_shared<opset1::Result>(conv)};
        function = std::make_shared<Function>(results, params, "convolution");
    }

private:
    std::vector<float> preAllocMemIn, preAllocMemOut;
};

TEST_P(ConvolutionLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    checkOffsetPadding(executableNetwork, inOffsetPadding, outOffsetPadding);
    CheckCPUImpl(executableNetwork, "Convolution", true);
}

// Common convolution params
ngraph::op::PadType padType{op::PadType::EXPLICIT};
const SizeVector numOutChannels = {32};
const size_t inOffsetPadding = 100, outOffsetPadding = 200;

namespace conv2D {
    // Common 2D convolution params
    const std::vector<SizeVector> kernels2d = {{3, 3}, {1, 1}, {3, 5}};
    const std::vector<SizeVector> strides2d = {{1, 1}, {2, 2}};
    const std::vector<std::vector<ptrdiff_t>> padBegins2d = {{0, 0}, {1, 1}};
    const std::vector<std::vector<ptrdiff_t>> padEnds2d =   {{0, 0}, {1, 1}};
    const std::vector<SizeVector> dilations2d = {{1, 1}, {2, 2}};
    const std::vector<SizeVector> plnInDims2d{{3, 5, 10, 10}}, specPlnInDims2d{{3, 1, 10, 10}, {3, 3, 10, 10}}, blkInDims2d{{3, 20, 10, 10}};

    const auto convSpecificParamsPln = ::testing::Combine(::testing::ValuesIn(kernels2d),
                                                          ::testing::ValuesIn(strides2d),
                                                          ::testing::ValuesIn(padBegins2d),
                                                          ::testing::ValuesIn(padEnds2d),
                                                          ::testing::ValuesIn(dilations2d),
                                                          ::testing::ValuesIn(numOutChannels),
                                                          ::testing::Values(padType));
    const auto convDefaultParamsPln = ::testing::Combine(convSpecificParamsPln,
                                                         ::testing::Values(Precision::FP32),
                                                         ::testing::Values(Precision::UNSPECIFIED),
                                                         ::testing::Values(Precision::UNSPECIFIED),
                                                         ::testing::Values(Layout::ANY),
                                                         ::testing::Values(Layout::ANY),
                                                         ::testing::ValuesIn(plnInDims2d),
                                                         ::testing::Values(CommonTestUtils::DEVICE_CPU));
    const std::vector<CPUSpecificParams> CPUParams_Conv2D_DefaulPln = { conv_ref_2D, conv_gemm_2D };
    INSTANTIATE_TEST_CASE_P(smoke_Conv2D_DefaulPln, ConvolutionLayerCPUTest,
                            ::testing::Combine(
                                convDefaultParamsPln,
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Conv2D_DefaulPln)),
                                ::testing::Values(inOffsetPadding),
                                ::testing::Values(outOffsetPadding)),
                            ConvolutionLayerCPUTest::getTestCaseName);

    const auto convSpecialParamsPln = ::testing::Combine(convSpecificParamsPln,
                                                         ::testing::Values(Precision::FP32),
                                                         ::testing::Values(Precision::UNSPECIFIED),
                                                         ::testing::Values(Precision::UNSPECIFIED),
                                                         ::testing::Values(Layout::ANY),
                                                         ::testing::Values(Layout::ANY),
                                                         ::testing::ValuesIn(specPlnInDims2d),
                                                         ::testing::Values(CommonTestUtils::DEVICE_CPU));
    const std::vector<CPUSpecificParams> CPUParams_Conv2D_SpecialPln = { conv_avx2_2D_IC_1_3_OC_8, conv_avx512_2D_IC_1_3_OC_16 };
    INSTANTIATE_TEST_CASE_P(smoke_Conv2D_SpecialPln, ConvolutionLayerCPUTest,
                            ::testing::Combine(
                                convSpecialParamsPln,
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Conv2D_SpecialPln)),
                                ::testing::Values(inOffsetPadding),
                                ::testing::Values(outOffsetPadding)),
                            ConvolutionLayerCPUTest::getTestCaseName);

    const auto convSpecificParamsBlk = ::testing::Combine(::testing::ValuesIn(kernels2d),
                                                          ::testing::ValuesIn(strides2d),
                                                          ::testing::ValuesIn(padBegins2d),
                                                          ::testing::ValuesIn(padEnds2d),
                                                          ::testing::ValuesIn(dilations2d),
                                                          ::testing::ValuesIn(numOutChannels),
                                                          ::testing::Values(padType));
    const auto convParamsBlk = ::testing::Combine(convSpecificParamsBlk,
                                                  ::testing::Values(Precision::FP32),
                                                  ::testing::Values(Precision::UNSPECIFIED),
                                                  ::testing::Values(Precision::UNSPECIFIED),
                                                  ::testing::Values(Layout::ANY),
                                                  ::testing::Values(Layout::ANY),
                                                  ::testing::ValuesIn(blkInDims2d),
                                                  ::testing::Values(CommonTestUtils::DEVICE_CPU));
    const std::vector<CPUSpecificParams> CPUParams_Conv2D_Blk = { conv_sse42_2D, conv_avx2_2D, conv_avx512_2D };
    INSTANTIATE_TEST_CASE_P(smoke_Conv2D_Blk, ConvolutionLayerCPUTest,
                            ::testing::Combine(
                                convParamsBlk,
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Conv2D_Blk)),
                                ::testing::Values(inOffsetPadding),
                                ::testing::Values(outOffsetPadding)),
                            ConvolutionLayerCPUTest::getTestCaseName);
} // namespace conv2D

namespace conv3D {
    // Common 3D convolution params
    const std::vector<SizeVector> kernels3d = {{3, 3, 3}, {1, 1, 1}, {3, 5, 5}};
    const std::vector<SizeVector> strides3d = {{1, 1, 1}, {2, 2, 2}};
    const std::vector<std::vector<ptrdiff_t>> padBegins3d = {{0, 0, 0}, {1, 1, 1}};
    const std::vector<std::vector<ptrdiff_t>> padEnds3d = {{0, 0, 0}/*, {1, 0, 0}*/}; // failed avx2 K(1.1.1)_PB(0.0.0)_PE(1.0.0)
    const std::vector<SizeVector> dilations3d = {{1, 1, 1}, {2, 2, 2}};
    const std::vector<SizeVector> plnInDims3d{{3, 5, 10, 10, 10}}, specPlnInDims3d{{3, 1, 10, 10, 10}, {3, 3, 10, 10, 10}}, blkInChn3d{{3, 20, 10, 10, 10}};

    const auto convSpecificParamsPln = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                          ::testing::ValuesIn(strides3d),
                                                          ::testing::ValuesIn(padBegins3d),
                                                          ::testing::ValuesIn(padEnds3d),
                                                          ::testing::ValuesIn(dilations3d),
                                                          ::testing::ValuesIn(numOutChannels),
                                                          ::testing::Values(padType));
    const auto convDefaultParamsPln = ::testing::Combine(convSpecificParamsPln,
                                                         ::testing::Values(Precision::FP32),
                                                         ::testing::Values(Precision::UNSPECIFIED),
                                                         ::testing::Values(Precision::UNSPECIFIED),
                                                         ::testing::Values(Layout::ANY),
                                                         ::testing::Values(Layout::ANY),
                                                         ::testing::ValuesIn(plnInDims3d),
                                                         ::testing::Values(CommonTestUtils::DEVICE_CPU));
    const std::vector<CPUSpecificParams> CPUParams_Conv3D_DefaulPln = { conv_ref_3D, conv_gemm_3D };
    INSTANTIATE_TEST_CASE_P(smoke_Conv3D_DefaulPln, ConvolutionLayerCPUTest,
                            ::testing::Combine(
                                convDefaultParamsPln,
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Conv3D_DefaulPln)),
                                ::testing::Values(inOffsetPadding),
                                ::testing::Values(outOffsetPadding)),
                            ConvolutionLayerCPUTest::getTestCaseName);

    const auto convSpecialParamsPln = ::testing::Combine(convSpecificParamsPln,
                                                         ::testing::Values(Precision::FP32),
                                                         ::testing::Values(Precision::UNSPECIFIED),
                                                         ::testing::Values(Precision::UNSPECIFIED),
                                                         ::testing::Values(Layout::ANY),
                                                         ::testing::Values(Layout::ANY),
                                                         ::testing::ValuesIn(specPlnInDims3d),
                                                         ::testing::Values(CommonTestUtils::DEVICE_CPU));
    const std::vector<CPUSpecificParams> CPUParams_Conv3D_SpecialPln = { conv_avx2_3D_IC_1_3_OC_8, conv_avx512_3D_IC_1_3_OC_16 };
    INSTANTIATE_TEST_CASE_P(smoke_Conv3D_SpecialPln, ConvolutionLayerCPUTest,
                            ::testing::Combine(
                                convSpecialParamsPln,
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Conv3D_SpecialPln)),
                                ::testing::Values(inOffsetPadding),
                                ::testing::Values(outOffsetPadding)),
                            ConvolutionLayerCPUTest::getTestCaseName);

    const auto convSpecificParamsBlk = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                           ::testing::ValuesIn(strides3d),
                                                           ::testing::ValuesIn(padBegins3d),
                                                           ::testing::ValuesIn(padEnds3d),
                                                           ::testing::ValuesIn(dilations3d),
                                                           ::testing::ValuesIn(numOutChannels),
                                                           ::testing::Values(padType));
    const auto convParamsBlk = ::testing::Combine(convSpecificParamsBlk,
                                                   ::testing::Values(Precision::FP32),
                                                   ::testing::Values(Precision::UNSPECIFIED),
                                                   ::testing::Values(Precision::UNSPECIFIED),
                                                   ::testing::Values(Layout::ANY),
                                                   ::testing::Values(Layout::ANY),
                                                   ::testing::ValuesIn(blkInChn3d),
                                                   ::testing::Values(CommonTestUtils::DEVICE_CPU));
    const std::vector<CPUSpecificParams> CPUParams_Conv3D_Blk = { conv_avx2_3D, conv_avx512_3D };
    INSTANTIATE_TEST_CASE_P(smoke_Conv3D_Blk8, ConvolutionLayerCPUTest,
                            ::testing::Combine(
                                convParamsBlk,
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Conv3D_Blk)),
                                ::testing::Values(inOffsetPadding),
                                ::testing::Values(outOffsetPadding)),
                            ConvolutionLayerCPUTest::getTestCaseName);
} // namespace conv3D

} // namespace CPULayerTestsDefinitions
