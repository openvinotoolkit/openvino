// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/ops.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/read_ir/compare_results.hpp"

namespace LayerTestsDefinitions {

namespace {
void compare(const std::shared_ptr<ngraph::Node> node,
             const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expected,
             const std::vector<InferenceEngine::Blob::Ptr>& actual,
             float threshold, const std::vector<InferenceEngine::Blob::Ptr> &inputs) {
    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> types(expected.size());
    auto outputs = node->outputs();
    LayerTestsUtils::LayerTestsCommon::Compare(expected, actual, threshold);
}

//void compare(const std::shared_ptr<ngraph::op::v0::DetectionOutput> node,
//             const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expected,
//             const std::vector<InferenceEngine::Blob::Ptr>& actual,
//             float threshold) {
//    ASSERT_EQ(expected.size(), actual.front()->byteSize());
//
//    size_t expSize = 0;
//    size_t actSize = 0;
//
//    const auto &expectedBuffer = expected.data();
//    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual.front());
//    IE_ASSERT(memory);
//    const auto lockedMemory = memory->wmap();
//    const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();
//
//    const float *expBuf = reinterpret_cast<const float *>(expectedBuffer);
//    const float *actBuf = reinterpret_cast<const float *>(actualBuffer);
//    for (size_t i = 0; i < actual.front()->size(); i+=7) {
//        if (expBuf[i] == -1)
//            break;
//        expSize += 7;
//    }
//    for (size_t i = 0; i < actual.front()->size(); i+=7) {
//        if (actBuf[i] == -1)
//            break;
//        actSize += 7;
//    }
//    ASSERT_EQ(expSize, actSize);
//    LayerTestsUtils::LayerTestsCommon::Compare<float>(expBuf, actBuf, expSize, 1e-2f);
//}
//
//namespace Proposal {
//template <class T>
//void Compare(const T *expected, const T *actual, std::size_t size,
//             T threshold, const std::size_t output_index, size_t& num_selected_boxes) {
//    for (std::size_t i = 0; i < size; ++i) {
//        const auto &ref = expected[i];
//        const auto &res = actual[i];
//
//        // verify until first -1 appears in the 1st output.
//        if (output_index == 0 &&
//            CommonTestUtils::ie_abs(ref - static_cast<T>(-1)) <= threshold) {
//            // output0 shape = {x, 5}
//            // output1 shape = {x}
//            // setting the new_size for output1 verification
//            num_selected_boxes = i / 5;
//            return;
//        }
//
//        const auto absoluteDifference = CommonTestUtils::ie_abs(res - ref);
//        if (absoluteDifference <= threshold) {
//            continue;
//        }
//
//        const auto max = std::max(CommonTestUtils::ie_abs(res),
//                                  CommonTestUtils::ie_abs(ref));
//        float diff =
//                static_cast<float>(absoluteDifference) / static_cast<float>(max);
//        ASSERT_TRUE(max != 0 && (diff <= static_cast<float>(threshold)))
//                                    << "Relative comparison of values expected: " << ref
//                                    << " and actual: " << res << " at index " << i
//                                    << " with threshold " << threshold << " failed";
//    }
//}
//} // namespace Proposal
//
//void compare(const std::shared_ptr<ngraph::op::v4::Proposal> node,
//             const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
//             const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs,
//             float threshold) {
//    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >= 0 ; outputIndex--) {
//        const auto& expected = expectedOutputs[outputIndex];
//        const auto& actual = actualOutputs[outputIndex];
//
//        const auto &expectedBuffer = expected.second.data();
//        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
//        IE_ASSERT(memory);
//        const auto lockedMemory = memory->wmap();
//        const auto actualBuffer = lockedMemory.as<const uint8_t *>();
//
//        auto k =  static_cast<float>(expected.first.size()) / actual->getTensorDesc().getPrecision().size();
//        // W/A for int4, uint4
//        if (expected.first == ngraph::element::Type_t::u4 || expected.first == ngraph::element::Type_t::i4) {
//            k /= 2;
//        }
//        if (outputIndex == 2) {
//            if (expected.second.size() != k * actual->byteSize())
//                throw std::runtime_error("Expected and actual size 3rd output have different size");
//        }
//
//        const auto &precision = actual->getTensorDesc().getPrecision();
//        size_t size = expected.second.size() / (k * actual->getTensorDesc().getPrecision().size());
//        switch (precision) {
//            case InferenceEngine::Precision::FP32: {
//                switch (expected.first) {
//                    case ngraph::element::Type_t::f32:
//                        LayerTestsUtils::LayerTestsCommon::Compare(
//                                reinterpret_cast<const float *>(expectedBuffer),
//                                reinterpret_cast<const float *>(actualBuffer), size, 0);
//                        break;
//                    case ngraph::element::Type_t::f64:
//                        LayerTestsUtils::LayerTestsCommon::Compare(
//                                reinterpret_cast<const double *>(expectedBuffer),
//                                reinterpret_cast<const float *>(actualBuffer), size, 0);
//                        break;
//                    default:
//                        break;
//                }
//
//                const auto fBuffer = lockedMemory.as<const float *>();
//                for (int i = size; i < actual->size(); i++) {
//                    ASSERT_TRUE(fBuffer[i] == -1.f) << "Invalid default value: " << fBuffer[i] << " at index: " << i;
//                }
//                break;
//            }
//            case InferenceEngine::Precision::I32: {
//                switch (expected.first) {
//                    case ngraph::element::Type_t::i32:
//                        LayerTestsUtils::LayerTestsCommon::Compare(
//                                reinterpret_cast<const int32_t *>(expectedBuffer),
//                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
//                        break;
//                    case ngraph::element::Type_t::i64:
//                        LayerTestsUtils::LayerTestsCommon::Compare(
//                                reinterpret_cast<const int64_t *>(expectedBuffer),
//                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
//                        break;
//                    default:
//                        break;
//                }
//                const auto iBuffer = lockedMemory.as<const int *>();
//                for (int i = size; i < actual->size(); i++) {
//                    ASSERT_TRUE(iBuffer[i] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
//                }
//                break;
//            }
//            default:
//                FAIL() << "Comparator for " << precision << " precision isn't supported";
//        }
//    }
//}
//
//void compare(const std::shared_ptr<ngraph::op::v5::NonMaxSuppression> node,
//             const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
//             const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs,
//             float threshold) {
//    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >= 0 ; outputIndex--) {
//        const auto& expected = expectedOutputs[outputIndex];
//        const auto& actual = actualOutputs[outputIndex];
//
//        const auto &expectedBuffer = expected.second.data();
//        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
//        IE_ASSERT(memory);
//        const auto lockedMemory = memory->wmap();
//        const auto actualBuffer = lockedMemory.as<const uint8_t *>();
//
//        const auto &precision = actual->getTensorDesc().getPrecision();
//        size_t size = expected.second.size() / (actual->getTensorDesc().getPrecision().size());
//        switch (precision) {
//            case InferenceEngine::Precision::FP32: {
//                switch (expected.first) {
//                    case ngraph::element::Type_t::f32:
//                        LayerTestsUtils::LayerTestsCommon::Compare(
//                                reinterpret_cast<const float *>(expectedBuffer),
//                                reinterpret_cast<const float *>(actualBuffer), size, 0);
//                        break;
//                    case ngraph::element::Type_t::f64:
//                        LayerTestsUtils::LayerTestsCommon::Compare(
//                                reinterpret_cast<const double *>(expectedBuffer),
//                                reinterpret_cast<const float *>(actualBuffer), size, 0);
//                        break;
//                    default:
//                        break;
//                }
//
//                const auto fBuffer = lockedMemory.as<const float *>();
//                for (int i = size; i < actual->size(); i++) {
//                    ASSERT_TRUE(fBuffer[i] == -1.f) << "Invalid default value: " << fBuffer[i] << " at index: " << i;
//                }
//                break;
//            }
//            case InferenceEngine::Precision::I32: {
//                switch (expected.first) {
//                    case ngraph::element::Type_t::i32:
//                        LayerTestsUtils::LayerTestsCommon::Compare(
//                                reinterpret_cast<const int32_t *>(expectedBuffer),
//                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
//                        break;
//                    case ngraph::element::Type_t::i64:
//                        LayerTestsUtils::LayerTestsCommon::Compare(
//                                reinterpret_cast<const int64_t *>(expectedBuffer),
//                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
//                        break;
//                    default:
//                        break;
//                }
//                const auto iBuffer = lockedMemory.as<const int *>();
//                for (int i = size; i < actual->size(); i++) {
//                    ASSERT_TRUE(iBuffer[i] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
//                }
//                break;
//            }
//            default:
//                FAIL() << "Comparator for " << precision << " precision isn't supported";
//        }
//    }
//}


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


//
void compare(const std::shared_ptr<ngraph::op::v5::NonMaxSuppression> node,
             const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
             const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs,
             float threshold, const std::vector<InferenceEngine::Blob::Ptr> &inputs) {
    size_t numBatches, numBoxes, numClasses;
    auto inShapeParams = node->get_input_shape(1);
    numBatches = inShapeParams[0];
    numClasses = inShapeParams[1];
    numBoxes = inShapeParams[2];
//        auto input0 = node->get_input_tensor(0);

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

//         Get input bboxes' coords
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

//        // Get actual bboxes' index/score
    std::vector<Box> actualList;
    {
        size_t selected_indices_size = actualOutputs[0]->byteSize() / sizeof(float);
        auto selected_indices_memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actualOutputs[0]);
        IE_ASSERT(selected_indices_memory);
        const auto selected_indices_lockedMemory = selected_indices_memory->rmap();
        const auto selected_indices_data = selected_indices_lockedMemory.as<const int32_t *>();

        auto selected_scores_memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actualOutputs[1]);
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

template<typename T>
void compareResults(const std::shared_ptr<ngraph::Node> node,
             const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expected,
             const std::vector<InferenceEngine::Blob::Ptr>& actual,
             float threshold, const std::vector<InferenceEngine::Blob::Ptr> &inputs) {
    return compare(ngraph::as_type_ptr<T>(node), expected, actual, threshold, inputs);
}
} // namespace

CompareMap getCompareMap() {
    CompareMap compareMap{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, compareResults<NAMESPACE::NAME>},
#include "ngraph/opsets/opset1_tbl.hpp"
#include "ngraph/opsets/opset2_tbl.hpp"
#include "ngraph/opsets/opset3_tbl.hpp"
#include "ngraph/opsets/opset4_tbl.hpp"
#include "ngraph/opsets/opset5_tbl.hpp"
#include "ngraph/opsets/opset6_tbl.hpp"
#undef NGRAPH_OP
    };
    return compareMap;
}

} // namespace LayerTestsDefinitions
