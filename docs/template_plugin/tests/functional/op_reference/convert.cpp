// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#include "transformations/utils/utils.hpp"
// #include "ngraph/runtime/reference/convert.hpp"
// #include "ngraph/runtime/tensor.hpp"
// #include "runtime/backend.hpp"
// #include "util/all_close.hpp"
// #include "util/all_close_f.hpp"
// #include "util/engine/test_engines.hpp"
// #include "util/ndarray.hpp"
// #include "util/test_case.hpp"
// #include "util/test_control.hpp"
// #include "util/test_tools.hpp"

using namespace ngraph;
using namespace InferenceEngine;

template <class T>
InferenceEngine::Blob::Ptr create_blob(const ngraph::element::Type& element_type, std::vector<T> values, size_t size = 0) {
    size_t real_size = size ? size : values.size() * sizeof(T) / element_type.size();
    auto blob = make_blob_with_precision(
        InferenceEngine::TensorDesc(InferenceEngine::details::convertPrecision(element_type), {real_size}, InferenceEngine::Layout::C));
    blob->allocate();
    MemoryBlob::Ptr minput = as<MemoryBlob>(blob);
    IE_ASSERT(minput);
    auto minputHolder = minput->wmap();

    std::memcpy(minputHolder.as<void*>(), values.data(), sizeof(T) * values.size());

    return blob;
}

struct ConvertParams {
    template <class IT, class OT>
    ConvertParams(const ngraph::PartialShape& shape, const ngraph::element::Type& iType, const ngraph::element::Type& oType, const std::vector<IT> iValues,
                  const std::vector<OT> oValues, size_t iSize = 0, size_t oSize = 0) {
        pshape = shape;
        inType = iType;
        outType = oType;

        inputData = create_blob(iType, iValues, iSize);
        refData = create_blob(oType, oValues, oSize);
    }
    ngraph::PartialShape pshape;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData;
    InferenceEngine::Blob::Ptr refData;
};

class CommonReferenceTest {
public:
    CommonReferenceTest(): targetDevice("TEMPLATE") {
        core = PluginCache::get().ie(targetDevice);
    }

    void Exec() {
        LoadNetwork();
        FillInputs();
        Infer();
        Compare();
    }

    void LoadNetwork() {
        InferenceEngine::CNNNetwork cnnNetwork(function);
        auto inputInfo = cnnNetwork.getInputsInfo();
        auto outputInfo = cnnNetwork.getOutputsInfo();
        for (const auto& param : function->get_parameters()) {
            inputInfo[param->get_friendly_name()]->setPrecision(InferenceEngine::details::convertPrecision(param->get_element_type()));
        }
        for (const auto& result : function->get_results()) {
            outputInfo[ngraph::op::util::create_ie_output_name(result->input_value(0))]->setPrecision(
                InferenceEngine::details::convertPrecision(result->get_element_type()));
        }
        executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice);
    }

    void FillInputs() {
        const auto& inputInfo = executableNetwork.GetInputsInfo();
        const auto& params = function->get_parameters();
        ASSERT_EQ(params.size(), inputData.size());
        ASSERT_EQ(inputInfo.size(), inputData.size());

        for (size_t i = 0; i < params.size(); i++) {
            const auto& param = params[i];
            const auto infoIt = inputInfo.find(param->get_friendly_name());
            GTEST_ASSERT_NE(infoIt, inputInfo.cend());

            const auto& info = infoIt->second;
            auto blob = make_blob_with_precision(info->getTensorDesc());
            blob->allocate();

            ASSERT_EQ(blob->byteSize(), inputData[i]->byteSize());

            MemoryBlob::Ptr mInputData = as<MemoryBlob>(inputData[i]);
            ASSERT_NE(mInputData, nullptr);
            auto minputDataHolder = mInputData->rmap();

            MemoryBlob::Ptr mBlob = as<MemoryBlob>(blob);
            ASSERT_NE(mBlob, nullptr);
            auto mBlobHolder = mBlob->wmap();

            std::memcpy(mBlobHolder.as<void*>(), minputDataHolder.as<const void*>(), inputData[i]->byteSize());
            inputData[i] = blob;
        }
    }

    void Infer() {
        inferRequest = executableNetwork.CreateInferRequest();

        const auto& inputsInfo = executableNetwork.GetInputsInfo();
        const auto& functionParams = function->get_parameters();
        for (int i = 0; i < functionParams.size(); ++i) {
            const auto& param = functionParams[i];
            const auto infoIt = inputsInfo.find(param->get_friendly_name());
            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

            const auto& info = infoIt->second;
            auto blob = inputData[i];

            inferRequest.SetBlob(info->name(), blob);
        }
        inferRequest.Infer();
    }

    void Compare() {
        ASSERT_EQ(executableNetwork.GetOutputsInfo().size(), refOutData.size());
        std::vector<InferenceEngine::Blob::Ptr> outputs;
        for (const auto& result : function->get_results()) {
            auto name = ngraph::op::util::create_ie_output_name(result->input_value(0));
            outputs.emplace_back(inferRequest.GetBlob(name));
        }

        ASSERT_EQ(refOutData.size(), outputs.size());
        for (size_t i = 0; i < refOutData.size(); i++) {
            CompareBlobs(refOutData[i], outputs[i]);
        }
    }

protected:
    const std::string targetDevice;
    std::shared_ptr<InferenceEngine::Core> core;
    std::shared_ptr<ngraph::Function> function;

    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest inferRequest;
    std::vector<InferenceEngine::Blob::Ptr> inputData;
    std::vector<InferenceEngine::Blob::Ptr> refOutData;
    float threshold = 1e-2f;

private:
    void CompareBlobs(const InferenceEngine::Blob::Ptr& refBlob, const InferenceEngine::Blob::Ptr& outBlob) {
        ASSERT_TRUE(refBlob != nullptr);
        ASSERT_TRUE(outBlob != nullptr);
        ASSERT_EQ(refBlob->getTensorDesc().getPrecision(), outBlob->getTensorDesc().getPrecision());
        ASSERT_EQ(refBlob->byteSize(), outBlob->byteSize());

        auto mRef = as<InferenceEngine::MemoryBlob>(refBlob);
        IE_ASSERT(mRef);
        const auto refLockMemory = mRef->rmap();
        const auto refBuffer = refLockMemory.as<const std::uint8_t*>();

        auto mOut = as<InferenceEngine::MemoryBlob>(outBlob);
        IE_ASSERT(mOut);
        const auto outLockMemory = mOut->rmap();
        const auto outBuffer = outLockMemory.as<const std::uint8_t*>();

        const auto& precision = refBlob->getTensorDesc().getPrecision();
        switch (precision) {
        case InferenceEngine::Precision::FP32:
            LayerTestsUtils::LayerTestsCommon::Compare<float, float>(reinterpret_cast<const float*>(refBuffer), reinterpret_cast<const float*>(outBuffer),
                                                                     refBlob->size(), threshold);
            break;
        case InferenceEngine::Precision::I32:
            LayerTestsUtils::LayerTestsCommon::Compare<int32_t, int32_t>(reinterpret_cast<const int32_t*>(refBuffer),
                                                                         reinterpret_cast<const int32_t*>(outBuffer), refBlob->size(), threshold);
            break;
        case InferenceEngine::Precision::I64:
            LayerTestsUtils::LayerTestsCommon::Compare<int64_t, int64_t>(reinterpret_cast<const int64_t*>(refBuffer),
                                                                         reinterpret_cast<const int64_t*>(outBuffer), refBlob->size(), threshold);
            break;
        case InferenceEngine::Precision::I8:
            LayerTestsUtils::LayerTestsCommon::Compare<int8_t, int8_t>(reinterpret_cast<const int8_t*>(refBuffer), reinterpret_cast<const int8_t*>(outBuffer),
                                                                       refBlob->size(), threshold);
            break;
        case InferenceEngine::Precision::U16:
            LayerTestsUtils::LayerTestsCommon::Compare<uint16_t, uint16_t>(reinterpret_cast<const uint16_t*>(refBuffer),
                                                                           reinterpret_cast<const uint16_t*>(outBuffer), refBlob->size(), threshold);
            break;
        case InferenceEngine::Precision::I16:
            LayerTestsUtils::LayerTestsCommon::Compare<int16_t, int16_t>(reinterpret_cast<const int16_t*>(refBuffer),
                                                                         reinterpret_cast<const int16_t*>(outBuffer), refBlob->size(), threshold);
            break;
        case InferenceEngine::Precision::BOOL:
        case InferenceEngine::Precision::U8:
            LayerTestsUtils::LayerTestsCommon::Compare<uint8_t, uint8_t>(reinterpret_cast<const uint8_t*>(refBuffer),
                                                                         reinterpret_cast<const uint8_t*>(outBuffer), refBlob->size(), threshold);
            break;
        case InferenceEngine::Precision::U64:
            LayerTestsUtils::LayerTestsCommon::Compare<uint64_t, uint64_t>(reinterpret_cast<const uint64_t*>(refBuffer),
                                                                           reinterpret_cast<const uint64_t*>(outBuffer), refBlob->size(), threshold);
            break;
        case InferenceEngine::Precision::BF16:
            LayerTestsUtils::LayerTestsCommon::Compare<ngraph::bfloat16, ngraph::bfloat16>(
                reinterpret_cast<const ngraph::bfloat16*>(refBuffer), reinterpret_cast<const ngraph::bfloat16*>(outBuffer), refBlob->size(), threshold);
            break;
        case InferenceEngine::Precision::FP16:
            LayerTestsUtils::LayerTestsCommon::Compare<ngraph::float16, ngraph::float16>(
                reinterpret_cast<const ngraph::float16*>(refBuffer), reinterpret_cast<const ngraph::float16*>(outBuffer), refBlob->size(), threshold);
            break;
        case InferenceEngine::Precision::I4:
        case InferenceEngine::Precision::U4:
            LayerTestsUtils::LayerTestsCommon::Compare<uint8_t, uint8_t>(reinterpret_cast<const uint8_t*>(refBuffer),
                                                                         reinterpret_cast<const uint8_t*>(outBuffer), refBlob->size() / 2, threshold);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
        }
    }
};

class ReferenceConvertLayerTest : public testing::TestWithParam<ConvertParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type,
                                                    const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto convert = std::make_shared<op::Convert>(in, expected_output_type);
        return std::make_shared<Function>(NodeVector {convert}, ParameterVector {in});
    }
};

TEST_P(ReferenceConvertLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Convert_With_Hardcided_Refs, ReferenceConvertLayerTest,
    ::testing::Values(
        // destination boolean
        ConvertParams(ngraph::PartialShape {2, 3}, ngraph::element::u8, ngraph::element::boolean,
                      std::vector<uint8_t> {0, 12, 23, 0, std::numeric_limits<uint8_t>::lowest(), std::numeric_limits<uint8_t>::max()},
                      std::vector<char> {0, 1, 1, 0, 0, 1}),
        ConvertParams(ngraph::PartialShape {2, 3}, ngraph::element::i32, ngraph::element::boolean,
                      std::vector<int32_t> {0, -12, 23, 0, std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max()},
                      std::vector<char> {0, 1, 1, 0, 1, 1}),
        ConvertParams(ngraph::PartialShape {3, 3}, ngraph::element::f32, ngraph::element::boolean,
                      std::vector<float> {0.f, 1.5745f, 0.12352f, 0.f, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(),
                                          std::numeric_limits<float>::min(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
                      std::vector<char> {0, 1, 1, 0, 1, 1, 1, 1, 1}),

        // destination i4
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u1, ngraph::element::i4, std::vector<uint8_t> {0xA0}, std::vector<uint8_t> {0x10, 0x10}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u4, ngraph::element::i4, std::vector<uint8_t> {0x12, 0x03}, std::vector<uint8_t> {0x12, 0x03},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u8, ngraph::element::i4, std::vector<uint8_t> {1, 2, 0, 3}, std::vector<uint8_t> {0x12, 0x03},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u16, ngraph::element::i4, std::vector<uint16_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {0x12, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u32, ngraph::element::i4, std::vector<uint32_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {0x12, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u64, ngraph::element::i4, std::vector<uint64_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {0x12, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i4, ngraph::element::i4, std::vector<uint8_t> {0xFE, 0x03}, std::vector<uint8_t> {0xFE, 0x03},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i8, ngraph::element::i4, std::vector<int8_t> {-1, -2, 2, 3}, std::vector<uint8_t> {0xFE, 0x23},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i16, ngraph::element::i4, std::vector<int16_t> {-1, -2, 2, 3},
                      std::vector<uint8_t> {0xFE, 0x23}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i32, ngraph::element::i4, std::vector<int32_t> {-1, -2, 2, 3},
                      std::vector<uint8_t> {0xFE, 0x23}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i64, ngraph::element::i4, std::vector<int64_t> {-1, -2, 2, 3},
                      std::vector<uint8_t> {0xFE, 0x23}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f16, ngraph::element::i4, std::vector<ngraph::float16> {-1, -2, 0, 3},
                      std::vector<uint8_t> {0xFE, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::bf16, ngraph::element::i4, std::vector<ngraph::bfloat16> {-1, -2, 0, 3},
                      std::vector<uint8_t> {0xFE, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f32, ngraph::element::i4, std::vector<float> {-1, -2, 2, 3}, std::vector<uint8_t> {0xFE, 0x23},
                      4, 4)));
