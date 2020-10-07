// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/set_blob.hpp>
#include "../test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph;
using namespace details;

namespace CPUBehaviorTestsDefinitions {

using NotDefaultInputOutputCPUTestParamsSet = std::tuple<SizeVector,                         // input shape
                                                         cpu_memory_format_t,                // input memomry format
                                                         cpu_memory_format_t,                // output memory format
                                                         size_t,                             // input offset padding
                                                         size_t,                             // output offset padding
                                                         Precision,                          // input precision
                                                         Precision,                          // output precision
                                                         BehaviorTestsDefinitions::setType>; // input/both

class NotDefaultInputOutputCPUTest : public testing::WithParamInterface<NotDefaultInputOutputCPUTestParamsSet>,
                                     public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NotDefaultInputOutputCPUTestParamsSet> obj) {
        SizeVector inShape;
        cpu_memory_format_t inMemFormat, outMemFormat;
        size_t inOffsetPadding, outOffsetPadding;
        Precision inPrec, outPrec;
        BehaviorTestsDefinitions::setType type;

        std::tie(inShape, inMemFormat, outMemFormat, inOffsetPadding, outOffsetPadding, inPrec, outPrec, type) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inShape) << "_";
        result << "inFmts=" << fmts2str(std::vector<cpu_memory_format_t>{inMemFormat}) << "_";
        result << "inOffsetPadding=" << inOffsetPadding << "_";
        result << "outFmts=" << fmts2str(std::vector<cpu_memory_format_t>{outMemFormat}) << "_";
        result << "outOffsetPadding=" << outOffsetPadding << "_";
        result << "inPrc=" << inPrec.name() << "_";
        result << "outPrc=" << outPrec.name() << "_";
        result << type;

        return result.str();
    }

    void ConfigureNetwork() override {
        for (const auto &in : cnnNetwork.getInputsInfo()) {
            TensorDesc descIn = createTensorDesc(in.second->getTensorDesc().getDims(), inPrc, inFmts[0], inOffsetPadding);
            in.second->setInputData(std::make_shared<Data>(in.second->name(), descIn));
        }

        const auto &outInfo = cnnNetwork.getOutputsInfo();
        auto out = outInfo.begin();
        TensorDesc descOut = createTensorDesc(out->second->getTensorDesc().getDims(), outPrc, outFmts[0], outOffsetPadding);
        *(out->second) = *std::make_shared<Data>(out->second->getName(), descOut);
    }

    inline void fillBlob(Blob::Ptr &blob) {
        switch (blob->getTensorDesc().getPrecision()) {
    #define CASE(X) case X: CommonTestUtils::fill_data_random<X>(blob); break;
            CASE(InferenceEngine::Precision::FP32)
            CASE(InferenceEngine::Precision::U8)
            CASE(InferenceEngine::Precision::U16)
            CASE(InferenceEngine::Precision::I8)
            CASE(InferenceEngine::Precision::I16)
            CASE(InferenceEngine::Precision::I64)
            CASE(InferenceEngine::Precision::U64)
            CASE(InferenceEngine::Precision::I32)
            CASE(InferenceEngine::Precision::BOOL)
    #undef CASE
            default:
                THROW_IE_EXCEPTION << "Can't fill blob with precision: " << blob->getTensorDesc().getPrecision();
        }
    }

    void Infer() override {
        inferRequest = executableNetwork.CreateInferRequest();
        inputs.clear();

        // inputs creation
        const auto &inputInfo = executableNetwork.GetInputsInfo();
        auto in = inputInfo.begin();
        preAllocMemIn.resize(inputInfo.size());
        for (size_t i = 0; i < inputInfo.size(); i++) {
            auto inPrec = in->second->getTensorDesc().getPrecision();
            auto inBlockingDesc = in->second->getTensorDesc().getBlockingDesc();
            size_t sizeIn = 1;
            for (size_t i = 0; i < inBlockingDesc.getBlockDims().size(); i++) {
                sizeIn *= inBlockingDesc.getBlockDims()[i];
            }
            sizeIn += inBlockingDesc.getOffsetPadding();
            preAllocMemIn[i].resize(sizeIn * inPrec.size());
            std::shared_ptr<IAllocator> inAllocator = details::make_pre_allocator(preAllocMemIn[i].data(), sizeIn * inPrec.size());
            auto blobIn = make_blob_with_precision(in->second->getTensorDesc(), inAllocator);
            blobIn->allocate();
            fillBlob(blobIn);
            inferRequest.SetBlob(in->second->name(), blobIn);

            // ngraph inputs creation
            auto inDims = in->second->getTensorDesc().getDims();
            auto refTensorDesc = TensorDesc(inPrec, inDims, TensorDesc::getLayoutByDims(inDims));
            auto refBlob = make_blob_with_precision(inPrec, refTensorDesc);
            refBlob->allocate();
            const uint8_t *src = blobIn->cbuffer().as<const uint8_t *>();
            uint8_t *dst = refBlob->buffer().as<uint8_t *>();
            for (size_t i = 0; i < refBlob->size(); i++) {
                memcpy(&dst[i * inPrec.size()], &src[blobIn->getTensorDesc().offset(i) * inPrec.size()], inPrec.size());
            }
            inputs.push_back(refBlob);

            in++;
        }

        // outputs creation
        if (type == BehaviorTestsDefinitions::setType::BOTH) {
            const auto outputInfo = executableNetwork.GetOutputsInfo().begin()->second;
            auto outPrec = outputInfo->getPrecision();
            auto outBlockingDesc = outputInfo->getTensorDesc().getBlockingDesc();
            size_t sizeOut = 1;
            for (size_t i = 0; i < outBlockingDesc.getBlockDims().size(); i++) {
                sizeOut *= outBlockingDesc.getBlockDims()[i];
            }
            sizeOut += outBlockingDesc.getOffsetPadding();
            preAllocMemOut.resize(sizeOut * outPrec.size());
            std::shared_ptr<IAllocator> outAllocator = details::make_pre_allocator(preAllocMemOut.data(), sizeOut * outPrec.size());
            auto blobOut = make_blob_with_precision(outputInfo->getTensorDesc(), outAllocator);
            blobOut->allocate();
            inferRequest.SetBlob(outputInfo->getName(), blobOut);
        }

        inferRequest.Infer();
    }

    void Compare(const std::vector<std::uint8_t> &expected, const Blob::Ptr &actual) override {
        auto blobToCmp = make_blob_with_precision(actual->getTensorDesc().getPrecision(), actual->getTensorDesc());
        blobToCmp->allocate();
        const uint8_t *actualBuffer = actual->cbuffer().as<const uint8_t *>();
        uint8_t *cmpBuffer = blobToCmp->buffer().as<uint8_t *>();
        const auto size = blobToCmp->size();
        auto prec = actual->getTensorDesc().getPrecision();
        for (size_t i = 0; i < size; i++)
             memcpy(&cmpBuffer[i * prec.size()], &actualBuffer[actual->getTensorDesc().offset(i) * prec.size()], prec.size());

        LayerTestsCommon::Compare(prec, expected.data(), cmpBuffer, size, threshold);
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        SizeVector inShape;
        inFmts.resize(1); outFmts.resize(1);
        std::tie(inShape, inFmts[0], outFmts[0], inOffsetPadding, outOffsetPadding, inPrc, outPrc, type) = this->GetParam();

        auto inNgPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        auto params = builder::makeParams(inNgPrc, {inShape, inShape});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));

        auto add = builder::makeEltwise(paramOuts[0], paramOuts[1], helpers::EltwiseTypes::ADD);

        ResultVector results{std::make_shared<opset4::Result>(add)};
        function = std::make_shared<Function>(results, params, "NotDefaultInputOutput");
    }

private:
    std::vector<std::vector<uint8_t>> preAllocMemIn;
    std::vector<uint8_t> preAllocMemOut;
    size_t inOffsetPadding, outOffsetPadding;
    BehaviorTestsDefinitions::setType type;
};

TEST_P(NotDefaultInputOutputCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
}

const size_t inOffset = 7, outOffset = 20;
const std::vector<Precision> precisionSet = {Precision::FP32, /*Precision::I16,*/ Precision::U8, Precision::I8, Precision::U16, Precision::I32,
                                             Precision::I64, Precision::U64};
const std::vector<BehaviorTestsDefinitions::setType> setType = {BehaviorTestsDefinitions::setType::INPUT, BehaviorTestsDefinitions::setType::BOTH};

// 1D case
const SizeVector inShape1D{20};
const cpu_memory_format_t memFormat1D{x};
auto testCase1D = ::testing::Combine(::testing::Values(inShape1D),
                                     ::testing::Values(memFormat1D),
                                     ::testing::Values(memFormat1D),
                                     ::testing::Values(inOffset),
                                     ::testing::Values(outOffset),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(setType));
INSTANTIATE_TEST_CASE_P(smoke_NotDefaultInputOutputCPUTest_1D, NotDefaultInputOutputCPUTest, testCase1D, NotDefaultInputOutputCPUTest::getTestCaseName);

// 2D case
const SizeVector inShape2D{20, 6};
const cpu_memory_format_t memFormat2D{nc};
auto testCase2D = ::testing::Combine(::testing::Values(inShape2D),
                                     ::testing::Values(memFormat2D),
                                     ::testing::Values(memFormat2D),
                                     ::testing::Values(inOffset),
                                     ::testing::Values(outOffset),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(setType));
INSTANTIATE_TEST_CASE_P(smoke_NotDefaultInputOutputCPUTest_2D, NotDefaultInputOutputCPUTest, testCase2D, NotDefaultInputOutputCPUTest::getTestCaseName);

// 3D case
const SizeVector inShape3D{20, 5, 6};
const cpu_memory_format_t memFormat3D{tnc};
auto testCase3D = ::testing::Combine(::testing::Values(inShape3D),
                                     ::testing::Values(memFormat3D),
                                     ::testing::Values(memFormat3D),
                                     ::testing::Values(inOffset),
                                     ::testing::Values(outOffset),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(setType));
INSTANTIATE_TEST_CASE_P(smoke_NotDefaultInputOutputCPUTest_3D, NotDefaultInputOutputCPUTest, testCase3D, NotDefaultInputOutputCPUTest::getTestCaseName);

// 4D case
const SizeVector inShape4D{2, 20, 5, 6};
const std::vector<cpu_memory_format_t> memFormat4D{nchw, nChw8c, nChw16c, nhwc};
auto testCase4D = ::testing::Combine(::testing::Values(inShape4D),
                                     ::testing::ValuesIn(memFormat4D),
                                     ::testing::ValuesIn(memFormat4D),
                                     ::testing::Values(inOffset),
                                     ::testing::Values(outOffset),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(setType));
INSTANTIATE_TEST_CASE_P(smoke_NotDefaultInputOutputCPUTest_4D, NotDefaultInputOutputCPUTest, testCase4D, NotDefaultInputOutputCPUTest::getTestCaseName);

// 5D case
const SizeVector inShape5D{2, 20, 3, 5, 6};
const std::vector<cpu_memory_format_t> memFormat5D{ncdhw, nCdhw8c, nCdhw16c, ndhwc};
auto testCase5D = ::testing::Combine(::testing::Values(inShape5D),
                                     ::testing::ValuesIn(memFormat5D),
                                     ::testing::ValuesIn(memFormat5D),
                                     ::testing::Values(inOffset),
                                     ::testing::Values(outOffset),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(setType));
INSTANTIATE_TEST_CASE_P(smoke_NotDefaultInputOutputCPUTest_5D, NotDefaultInputOutputCPUTest, testCase5D, NotDefaultInputOutputCPUTest::getTestCaseName);

// 6D case
const SizeVector inShape6D{2, 20, 7, 3, 5, 6};
const cpu_memory_format_t memFormat6D{any};
auto testCase6D = ::testing::Combine(::testing::Values(inShape6D),
                                     ::testing::Values(memFormat6D),
                                     ::testing::Values(memFormat6D),
                                     ::testing::Values(inOffset),
                                     ::testing::Values(outOffset),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(precisionSet),
                                     ::testing::ValuesIn(setType));
INSTANTIATE_TEST_CASE_P(smoke_NotDefaultInputOutputCPUTest_6D, NotDefaultInputOutputCPUTest, testCase6D, NotDefaultInputOutputCPUTest::getTestCaseName);

} // namespace CPUBehaviorTestsDefinitions
