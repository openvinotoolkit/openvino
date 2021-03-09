// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include "shared_test_classes/single_layer/non_max_suppression.hpp"

namespace LayerTestsDefinitions {

using namespace ngraph;
using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

std::string NmsLayerTest::getTestCaseName(testing::TestParamInfo<NmsParams> obj) {
    InputShapeParams inShapeParams;
    InputPrecisions inPrecisions;
    int32_t maxOutBoxesPerClass;
    float iouThr, scoreThr, softNmsSigma;
    op::v5::NonMaxSuppression::BoxEncodingType boxEncoding;
    bool sortResDescend;
    element::Type outType;
    std::string targetDevice;
    std::tie(inShapeParams, inPrecisions, maxOutBoxesPerClass, iouThr, scoreThr, softNmsSigma, boxEncoding, sortResDescend, outType, targetDevice) = obj.param;

    size_t numBatches, numBoxes, numClasses;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

    Precision paramsPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

    std::ostringstream result;
    result << "numBatches=" << numBatches << "_numBoxes=" << numBoxes << "_numClasses=" << numClasses << "_";
    result << "paramsPrec=" << paramsPrec << "_maxBoxPrec=" << maxBoxPrec << "_thrPrec=" << thrPrec << "_";
    result << "maxOutBoxesPerClass=" << maxOutBoxesPerClass << "_";
    result << "iouThr=" << iouThr << "_scoreThr=" << scoreThr << "_softNmsSigma=" << softNmsSigma << "_";
    result << "boxEncoding=" << boxEncoding << "_sortResDescend=" << sortResDescend << "_outType=" << outType << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void NmsLayerTest::GenerateInputs() {
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

void NmsLayerTest::Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                           const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) {
    CompareBBoxes(expectedOutputs, actualOutputs);
}

void NmsLayerTest::CompareBuffer(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                                 const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) {
    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >=0 ; outputIndex--) {
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

                const auto fBuffer = lockedMemory.as<const float *>();
                for (int i = size; i < actual->size(); i++) {
                    ASSERT_TRUE(fBuffer[i] == -1.f) << "Invalid default value: " << fBuffer[i] << " at index: " << i;
                }
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
                const auto iBuffer = lockedMemory.as<const int *>();
                for (int i = size; i < actual->size(); i++) {
                    ASSERT_TRUE(iBuffer[i] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
                }
                break;
            }
            default:
                FAIL() << "Comparator for " << precision << " precision isn't supported";
        }
    }
}

typedef struct Rect {
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
} Rect;

class Box {
public:
    Box() = default;

    Box(int32_t batchId, int32_t classId, int32_t boxId, Rect rect, float score) {
        this->batchId = batchId;
        this->classId = classId;
        this->boxId = boxId;
        this->rect = rect;
        this->score = score;
    }

    int32_t batchId;
    int32_t classId;
    int32_t boxId;
    Rect rect;
    float score;
};

/*
 * 1: selected_indices - tensor of type T_IND and shape [number of selected boxes, 3] containing information about selected boxes as triplets 
 *    [batch_index, class_index, box_index].
 * 2: selected_scores - tensor of type T_THRESHOLDS and shape [number of selected boxes, 3] containing information about scores for each selected box as triplets
 *    [batch_index, class_index, box_score].
 * 3: valid_outputs - 1D tensor with 1 element of type T_IND representing the total number of selected boxes.
 */
void NmsLayerTest::CompareBBoxes(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                                 const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) {
    size_t numBatches, numBoxes, numClasses;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

    auto iouFunc = [](const Box& boxI, const Box& boxJ) {
        const Rect& rectI = boxI.rect;
        const Rect& rectJ = boxJ.rect;

        float areaI = (rectI.y2 - rectI.y1) * (rectI.x2 - rectI.x1);
        float areaJ = (rectJ.y2 - rectJ.y1) * (rectJ.x2 - rectJ.x1);

        if (areaI <= 0.0f || areaJ <= 0.0f) {
            return 0.0f;
        }

        float intersection_ymin = std::max(rectI.y1, rectJ.y1);
        float intersection_xmin = std::max(rectI.x1, rectJ.x1);
        float intersection_ymax = std::min(rectI.y2, rectJ.y2);
        float intersection_xmax = std::min(rectI.x2, rectJ.x2);

        float intersection_area =
            std::max(intersection_ymax - intersection_ymin, 0.0f) *
            std::max(intersection_xmax - intersection_xmin, 0.0f);

        return intersection_area / (areaI + areaJ - intersection_area);
    };

    // Get input bboxes' coords
    std::vector<std::vector<Rect>> coordList(numBatches, std::vector<Rect>(numBoxes));
    {
        const auto &input = inputs[0];
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->rmap();
        const auto buffer = lockedMemory.as<const float *>();
        for (size_t i = 0; i < numBatches; ++i) {
            for (size_t j = 0; j < numBoxes; ++j) {
                const int32_t y1 = static_cast<int32_t>(buffer[(i*numBoxes+j)*4+0]);
                const int32_t x1 = static_cast<int32_t>(buffer[(i*numBoxes+j)*4+1]);
                const int32_t y2 = static_cast<int32_t>(buffer[(i*numBoxes+j)*4+2]);
                const int32_t x2 = static_cast<int32_t>(buffer[(i*numBoxes+j)*4+3]);

                coordList[i][j] = { std::min(y1, y2),
                                    std::min(x1, x2),
                                    std::max(y1, y2),
                                    std::max(x1, x2) };
            }
        }
    }

    auto compareBox = [](const Box& boxA, const Box& boxB) {
        return (boxA.batchId < boxB.batchId) ||
                (boxA.batchId == boxB.batchId && boxA.classId < boxB.classId) ||
                (boxA.batchId == boxB.batchId && boxA.classId == boxB.classId && boxA.boxId < boxB.boxId);
    };

    // Get expected bboxes' index/score
    std::vector<Box> expectedList;
    {
        size_t selected_indices_size = expectedOutputs[0].second.size() / expectedOutputs[0].first.size();
        size_t selected_scores_size = expectedOutputs[1].second.size() / expectedOutputs[1].first.size();
        ASSERT_TRUE(selected_indices_size == selected_scores_size);

        expectedList.resize(selected_indices_size);

        if (expectedOutputs[0].first.size() == 4) {
            auto selected_indices_data = reinterpret_cast<const int32_t *>(expectedOutputs[0].second.data());

            for (size_t i = 0; i < selected_indices_size; i += 3) {
                expectedList[i/3].batchId = selected_indices_data[i+0];
                expectedList[i/3].classId = selected_indices_data[i+1];
                expectedList[i/3].boxId   = selected_indices_data[i+2];
                expectedList[i/3].rect    = coordList[expectedList[i/3].batchId][expectedList[i/3].boxId];
            }
        } else {
            auto selected_indices_data = reinterpret_cast<const int64_t *>(expectedOutputs[0].second.data());

            for (size_t i = 0; i < selected_indices_size; i += 3) {
                expectedList[i/3].batchId = static_cast<int32_t>(selected_indices_data[i+0]);
                expectedList[i/3].classId = static_cast<int32_t>(selected_indices_data[i+1]);
                expectedList[i/3].boxId   = static_cast<int32_t>(selected_indices_data[i+2]);
                expectedList[i/3].rect    = coordList[expectedList[i/3].batchId][expectedList[i/3].boxId];
            }
        }

        if (expectedOutputs[1].first.size() == 4) {
            auto selected_scores_data = reinterpret_cast<const float *>(expectedOutputs[0].second.data());
            for (size_t i = 0; i < selected_scores_size; i += 3) {
                expectedList[i/3].score = selected_scores_data[i+2];
            }
        } else {
            auto selected_scores_data = reinterpret_cast<const double *>(expectedOutputs[0].second.data());
            for (size_t i = 0; i < selected_scores_size; i += 3) {
                expectedList[i/3].score = static_cast<float>(selected_scores_data[i+2]);
            }
        }

        std::sort(expectedList.begin(), expectedList.end(), compareBox);
    }

    // Get actual bboxes' index/score
    std::vector<Box> actualList;
    {
        size_t selected_indices_size = actualOutputs[0]->byteSize() / sizeof(float);
        auto selected_indices_memory = as<MemoryBlob>(actualOutputs[0]);
        IE_ASSERT(selected_indices_memory);
        const auto selected_indices_lockedMemory = selected_indices_memory->rmap();
        const auto selected_indices_data = selected_indices_lockedMemory.as<const int32_t *>();

        auto selected_scores_memory = as<MemoryBlob>(actualOutputs[1]);
        IE_ASSERT(selected_scores_memory);
        const auto selected_scores_lockedMemory = selected_scores_memory->rmap();
        const auto selected_scores_data = selected_scores_lockedMemory.as<const float *>();

        for (size_t i = 0; i < selected_indices_size; i += 3) {
            const int32_t batchId = selected_indices_data[i+0];
            const int32_t classId = selected_indices_data[i+1];
            const int32_t boxId   = selected_indices_data[i+2];
            const float score = selected_scores_data[i+2];
            if (batchId == -1 || classId == -1 || boxId == -1)
                break;

            actualList.emplace_back(batchId, classId, boxId, coordList[batchId][boxId], score);
        }
        std::sort(actualList.begin(), actualList.end(), compareBox);
    }

    std::vector<Box> intersectionList;
    std::vector<Box> differenceList;
    {
        std::list<Box> tempExpectedList(expectedList.size()), tempActualList(actualList.size());
        std::copy(expectedList.begin(), expectedList.end(), tempExpectedList.begin());
        std::copy(actualList.begin(), actualList.end(), tempActualList.begin());
        auto sameBox = [](const Box& boxA, const Box& boxB) {
            return (boxA.batchId == boxB.batchId) && (boxA.classId == boxB.classId) && (boxA.boxId == boxB.boxId);
        };

        for (auto itA = tempActualList.begin(); itA != tempActualList.end(); ++itA) {
            bool found = false;
            for (auto itB = tempExpectedList.begin(); itB != tempExpectedList.end(); ++itB) {
                if (sameBox(*itA, *itB)) {
                    intersectionList.emplace_back(*itB);
                    tempExpectedList.erase(itB);
                    found = true;
                    break;
                }
            }

            if (!found) {
                differenceList.emplace_back(*itA);
            }
        }
        differenceList.insert(differenceList.end(), tempExpectedList.begin(), tempExpectedList.end());

        for (auto& item : differenceList) {
            if ((item.rect.x1 == item.rect.x2) || (item.rect.y1 == item.rect.y2))
                continue;

            float maxIou = 0.f;
            for (auto& refItem : intersectionList) {
                maxIou = std::max(maxIou, iouFunc(item, refItem));

                if (maxIou > 0.3f) break;
            }

            ASSERT_TRUE(maxIou > 0.3f) << "MaxIOU: " << maxIou
                << ", expectedList.size(): " << expectedList.size() << ", actualList.size(): " << actualList.size()
                << ", intersectionList.size(): " << intersectionList.size() << ", diffList.size(): " << differenceList.size()
                << ", batchId: " << item.batchId << ", classId: " << item.classId << ", boxId: " << item.boxId
                << ", score: " << item.score << ", coord: " << item.rect.x1 << ", " << item.rect.y1 << ", " << item.rect.x2 << ", " << item.rect.y2;
        }
    }
}

void NmsLayerTest::SetUp() {
    InputPrecisions inPrecisions;
    size_t maxOutBoxesPerClass;
    float iouThr, scoreThr, softNmsSigma;
    op::v5::NonMaxSuppression::BoxEncodingType boxEncoding;
    bool sortResDescend;
    element::Type outType;
    std::tie(inShapeParams, inPrecisions, maxOutBoxesPerClass, iouThr, scoreThr, softNmsSigma, boxEncoding, sortResDescend, outType,
             targetDevice) = this->GetParam();

    size_t numBatches, numBoxes, numClasses;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

    Precision paramsPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

    numOfSelectedBoxes = std::min(numBoxes, maxOutBoxesPerClass) * numBatches * numClasses;

    const std::vector<size_t> boxesShape{numBatches, numBoxes, 4}, scoresShape{numBatches, numClasses, numBoxes};
    auto ngPrc = convertIE2nGraphPrc(paramsPrec);
    auto params = builder::makeParams(ngPrc, {boxesShape, scoresShape});
    auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));

    auto nms = builder::makeNms(paramOuts[0], paramOuts[1], convertIE2nGraphPrc(maxBoxPrec), convertIE2nGraphPrc(thrPrec), maxOutBoxesPerClass, iouThr,
                                scoreThr, softNmsSigma, boxEncoding, sortResDescend, outType);
    auto nms_0_identity = std::make_shared<opset5::Multiply>(nms->output(0), opset5::Constant::create(outType, Shape{1}, {1}));
    auto nms_1_identity = std::make_shared<opset5::Multiply>(nms->output(1), opset5::Constant::create(ngPrc, Shape{1}, {1}));
    auto nms_2_identity = std::make_shared<opset5::Multiply>(nms->output(2), opset5::Constant::create(outType, Shape{1}, {1}));
    function = std::make_shared<Function>(OutputVector{nms_0_identity, nms_1_identity, nms_2_identity}, params, "NMS");
}

}  // namespace LayerTestsDefinitions
