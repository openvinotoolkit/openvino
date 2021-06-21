// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/matrix_nms.hpp"

namespace LayerTestsDefinitions {

using namespace ngraph;
using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

std::string MatrixNmsLayerTest::getTestCaseName(testing::TestParamInfo<NmsParams> obj) {
    InputShapeParams inShapeParams;
    InputPrecisions inPrecisions;
    op::v8::MatrixNms::SortResultType sortResultType;
    bool sortResultAcrossBatch;
    element::Type outType;
    int nmsTopK, keepTopK, backgroudClass;
    op::v8::MatrixNms::DecayFunction decayFunction;
    std::string targetDevice;
    std::tie(inShapeParams, inPrecisions, sortResultType, sortResultAcrossBatch, outType, nmsTopK, keepTopK,
        backgroudClass, decayFunction, targetDevice) = obj.param;

    size_t numBatches, numBoxes, numClasses;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

    Precision paramsPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

    std::ostringstream result;
    result << "numBatches=" << numBatches << "_numBoxes=" << numBoxes << "_numClasses=" << numClasses << "_";
    result << "paramsPrec=" << paramsPrec << "_maxBoxPrec=" << maxBoxPrec << "_thrPrec=" << thrPrec << "_";
    result << "sortResultType=" << sortResultType << "_sortResultAcrossBatch=" << sortResultAcrossBatch << "_";
    result << "outType=" << outType << "_nmsTopK=" << nmsTopK << "_keepTopK=" << keepTopK << "_";
    result << "backgroudClass=" << backgroudClass << "_decayFunction=" << decayFunction << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void MatrixNmsLayerTest::GenerateInputs() {
    size_t it = 0;
    for (const auto &input : cnnNetwork.getInputsInfo()) {
        const auto &info = input.second;
        Blob::Ptr blob;

        if (it == 1) {
            blob = make_blob_with_precision(info->getTensorDesc());
            blob->allocate();
            CommonTestUtils::fill_data_random_float<Precision::FP32>(blob, 1, 0, 1000);
        } else {
            blob = GenerateInput(*info);
        }
        inputs.push_back(blob);
        it++;
    }
}

void MatrixNmsLayerTest::Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                                     const std::vector<Blob::Ptr> &actualOutputs) {
    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >= 0 ; outputIndex--) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualOutputs[outputIndex];

        const auto &expectedBuffer = expected.second.data();
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto actualBuffer = lockedMemory.as<const uint8_t *>();

        auto k =  static_cast<float>(expected.first.size()) / actual->getTensorDesc().getPrecision().size();
        // W/A for int4, uint4
        if (expected.first == ngraph::element::Type_t::u4 || expected.first == ngraph::element::Type_t::i4) {
            k /= 2;
        }
        if (outputIndex == 2) {
            if (expected.second.size() != k * actual->byteSize())
                throw std::runtime_error("Expected and actual size 3rd output have different size");
        }

        const auto &precision = actual->getTensorDesc().getPrecision();
        size_t size = expected.second.size() / (k * actual->getTensorDesc().getPrecision().size());
        switch (precision) {
            case InferenceEngine::Precision::FP32: {
                switch (expected.first) {
                    case ngraph::element::Type_t::f32:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const float *>(expectedBuffer),
                                reinterpret_cast<const float *>(actualBuffer), size, 0);
                        break;
                    case ngraph::element::Type_t::f64:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const double *>(expectedBuffer),
                                reinterpret_cast<const float *>(actualBuffer), size, 0);
                        break;
                    default:
                        break;
                }

                // TODO: test usage only, dynamic shape do not need check the tail value
                //const auto fBuffer = lockedMemory.as<const float *>();
                //for (int i = size; i < actual->size(); i++) {
                //    ASSERT_TRUE(fBuffer[i] == -1.f) << "Invalid default value: " << fBuffer[i] << " at index: " << i;
                //}
                break;
            }
            case InferenceEngine::Precision::I32: {
                switch (expected.first) {
                    case ngraph::element::Type_t::i32:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const int32_t *>(expectedBuffer),
                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                        break;
                    case ngraph::element::Type_t::i64:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const int64_t *>(expectedBuffer),
                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                        break;
                    default:
                        break;
                }
                // TODO: test usage only, dynamic shape do not need check the tail value
                //const auto iBuffer = lockedMemory.as<const int *>();
                //for (int i = size; i < actual->size(); i++) {
                //    ASSERT_TRUE(iBuffer[i] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
                //}
                break;
            }
            default:
                FAIL() << "Comparator for " << precision << " precision isn't supported";
        }
    }
}

void MatrixNmsLayerTest::SetUp() {
    InputShapeParams inShapeParams;
    InputPrecisions inPrecisions;
    op::v8::MatrixNms::SortResultType sortResultType;
    bool sortResultAcrossBatch;
    element::Type outType;
    int nmsTopK, keepTopK, backgroudClass;
    op::v8::MatrixNms::DecayFunction decayFunction;
    std::tie(inShapeParams, inPrecisions, sortResultType, sortResultAcrossBatch, outType, nmsTopK, keepTopK,
        backgroudClass, decayFunction, targetDevice) = this->GetParam();

    size_t numBatches, numBoxes, numClasses;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

    Precision paramsPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

    const std::vector<size_t> boxesShape{numBatches, numBoxes, 4}, scoresShape{numBatches, numClasses, numBoxes};
    auto ngPrc = convertIE2nGraphPrc(paramsPrec);
    auto params = builder::makeParams(ngPrc, {boxesShape, scoresShape});
    auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));
    auto nms = std::make_shared<opset8::MatrixNms>(paramOuts[0], paramOuts[1], sortResultType, sortResultAcrossBatch, outType, 0.5f,
        nmsTopK, keepTopK, backgroudClass, decayFunction);
    auto nms_0_identity = std::make_shared<opset5::Multiply>(nms->output(0), opset5::Constant::create(element::f32, Shape{1}, {1}));
    auto nms_1_identity = std::make_shared<opset5::Multiply>(nms->output(1), opset5::Constant::create(outType, Shape{1}, {1}));
    auto nms_2_identity = std::make_shared<opset5::Multiply>(nms->output(2), opset5::Constant::create(outType, Shape{1}, {1}));
    function = std::make_shared<Function>(OutputVector{nms_0_identity, nms_1_identity, nms_2_identity}, params, "NMS");
}

}  // namespace LayerTestsDefinitions
