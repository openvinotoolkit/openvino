// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/nms_rotated.hpp"
#include "openvino/op/nms_rotated.hpp"

#include <vector>

namespace LayerTestsDefinitions {

using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

std::string NmsRotatedLayerTest::getTestCaseName(const testing::TestParamInfo<NmsRotatedParams>& obj) {
    InputShapeParams inShapeParams;
    InputPrecisions inPrecisions;
    int32_t maxOutBoxesPerClass;
    float iouThr, scoreThr;
    bool sortResDescend, clockwise;
    ov::element::Type outType;
    std::string targetDevice;
    std::tie(inShapeParams,
             inPrecisions,
             maxOutBoxesPerClass,
             iouThr,
             scoreThr,
             sortResDescend,
             outType,
             clockwise,
             targetDevice) = obj.param;

    size_t numBatches, numBoxes, numClasses;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

    Precision inputPrec, maxBoxPrec, thrPrec;
    std::tie(inputPrec, maxBoxPrec, thrPrec) = inPrecisions;

    std::ostringstream result;
    result << "numBatches=" << numBatches << "_numBoxes=" << numBoxes << "_numClasses=" << numClasses << "_";
    result << "inputPrec=" << inputPrec << "_maxBoxPrec=" << maxBoxPrec << "_thrPrec=" << thrPrec << "_";
    result << "maxOutBoxesPerClass=" << maxOutBoxesPerClass << "_";
    result << "iouThr=" << iouThr << "_scoreThr=" << scoreThr << "_";
    result << "sortResDescend=" << sortResDescend << "_outType=" << outType << "_";
    result << "clockwise=" << clockwise << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void NmsRotatedLayerTest::GenerateInputs() {
    size_t it = 0;
    for (const auto& input : cnnNetwork.getInputsInfo()) {
        const auto& info = input.second;
        Blob::Ptr blob;

        if (it == 1) {
            blob = make_blob_with_precision(info->getTensorDesc());
            blob->allocate();
            if (info->getTensorDesc().getPrecision() == Precision::FP32) {
                ov::test::utils::fill_data_random_float<InferenceEngine::Precision::FP32>(blob, 1, 0, 1000);
            } else {
                ov::test::utils::fill_data_random_float<InferenceEngine::Precision::FP16>(blob, 1, 0, 1000);
            }
        } else {
            blob = GenerateInput(*info);
        }
        inputs.push_back(blob);
        it++;
    }
}

void NmsRotatedLayerTest::Compare(
    const std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
    const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) {
    size_t num_batches, num_boxes, num_classes;
    std::tie(num_batches, num_boxes, num_classes) = inShapeParams;

    struct OutBox {
        OutBox() = default;

        OutBox(int32_t batchId, int32_t classId, int32_t boxId, float score) {
            this->batchId = batchId;
            this->classId = classId;
            this->boxId = boxId;
            this->score = score;
        }

        bool operator==(const OutBox& rhs) const {
            return batchId == rhs.batchId && classId == rhs.classId && boxId == rhs.boxId;
        }

        int32_t batchId;
        int32_t classId;
        int32_t boxId;
        float score;
    };

    std::vector<OutBox> expected;
    {
        const auto selected_indices_size = expectedOutputs[0].second.size() / expectedOutputs[0].first.size();
        const auto selected_scores_size = expectedOutputs[1].second.size() / expectedOutputs[1].first.size();

        ASSERT_EQ(selected_indices_size, selected_scores_size);

        const auto boxes_count = selected_indices_size / 3;
        expected.resize(boxes_count);

        if (expectedOutputs[0].first.size() == 4) {
            auto selected_indices_data = reinterpret_cast<const int32_t*>(expectedOutputs[0].second.data());

            for (size_t i = 0; i < selected_indices_size; i += 3) {
                expected[i / 3].batchId = selected_indices_data[i + 0];
                expected[i / 3].classId = selected_indices_data[i + 1];
                expected[i / 3].boxId = selected_indices_data[i + 2];
            }
        } else {
            auto selected_indices_data = reinterpret_cast<const int64_t*>(expectedOutputs[0].second.data());

            for (size_t i = 0; i < selected_indices_size; i += 3) {
                expected[i / 3].batchId = static_cast<int32_t>(selected_indices_data[i + 0]);
                expected[i / 3].classId = static_cast<int32_t>(selected_indices_data[i + 1]);
                expected[i / 3].boxId = static_cast<int32_t>(selected_indices_data[i + 2]);
            }
        }

         if (expectedOutputs[1].first.size() == 4) {
            auto selected_scores_data = reinterpret_cast<const float*>(expectedOutputs[1].second.data());
            for (size_t i = 0; i < selected_scores_size; i += 3) {
                expected[i / 3].score = selected_scores_data[i + 2];
            }
        } else {
            auto selected_scores_data = reinterpret_cast<const double*>(expectedOutputs[1].second.data());
            for (size_t i = 0; i < selected_scores_size; i += 3) {
                expected[i / 3].score = static_cast<float>(selected_scores_data[i + 2]);
            }
        }
    }

    std::vector<OutBox> actual;
    {
        const auto selected_indices_size = actualOutputs[0]->byteSize() / sizeof(float);
        const auto selected_indices_memory = as<MemoryBlob>(actualOutputs[0]);
        IE_ASSERT(selected_indices_memory);
        const auto selected_indices_lockedMemory = selected_indices_memory->rmap();
        const auto selected_indices_data = selected_indices_lockedMemory.as<const int32_t*>();

        const auto selected_scores_memory = as<MemoryBlob>(actualOutputs[1]);
        IE_ASSERT(selected_scores_memory);
        const auto selected_scores_lockedMemory = selected_scores_memory->rmap();
        const auto selected_scores_data = selected_scores_lockedMemory.as<const float*>();

        for (size_t i = 0; i < selected_indices_size; i += 3) {
            const int32_t batchId = selected_indices_data[i + 0];
            const int32_t classId = selected_indices_data[i + 1];
            const int32_t boxId = selected_indices_data[i + 2];
            const float score = selected_scores_data[i + 2];
            if (batchId == -1 || classId == -1 || boxId == -1)
                break;

            actual.emplace_back(batchId, classId, boxId, score);
        }
    }

    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(expected[i], actual[i]) << ", i=" << i;
        ASSERT_NEAR(expected[i].score, actual[i].score, abs_threshold) << ", i=" << i;
    }
}

void NmsRotatedLayerTest::SetUp() {
    InputPrecisions inPrecisions;
    size_t maxOutBoxesPerClass;
    float iouThr, scoreThr;
    bool sortResDescend, clockwise;
    ov::element::Type outType;
    std::tie(inShapeParams,
             inPrecisions,
             maxOutBoxesPerClass,
             iouThr,
             scoreThr,
             sortResDescend,
             outType,
             clockwise,
             targetDevice) = this->GetParam();

    size_t numBatches, numBoxes, numClasses;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

    Precision inputPrec, maxBoxPrec, thrPrec;
    std::tie(inputPrec, maxBoxPrec, thrPrec) = inPrecisions;

    if (inputPrec == Precision::FP16) {
        abs_threshold = 0.1;
    } else {
        abs_threshold = std::numeric_limits<float>::epsilon();
    }

    ov::ParameterVector params;

    const std::vector<size_t> boxesShape{numBatches, numBoxes, 5}, scoresShape{numBatches, numClasses, numBoxes};
    const auto ngPrc = convertIE2nGraphPrc(inputPrec);

    const auto boxesNode = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(boxesShape));
    params.push_back(boxesNode);
    const auto scoresNode = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(scoresShape));
    params.push_back(scoresNode);

    const auto maxOutputBoxesPerClassNode = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::u32,
                                                                                   ov::Shape{},
                                                                                   std::vector<size_t>{maxOutBoxesPerClass});
    const auto iouThresholdNode = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32,
                                                                                   ov::Shape{},
                                                                                   std::vector<float>{iouThr});
    const auto scoreTresholdNode = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32,
                                                                                   ov::Shape{},
                                                                                   std::vector<float>{scoreThr});

    const auto nmsNode = std::make_shared<ov::op::v13::NMSRotated>(params[0],
                                                               params[1],
                                                               maxOutputBoxesPerClassNode,
                                                               iouThresholdNode,
                                                               scoreTresholdNode,
                                                               sortResDescend,
                                                               outType,
                                                               clockwise);

    function = std::make_shared<ov::Model>(nmsNode, params, "NMS");
}
}  // namespace LayerTestsDefinitions
