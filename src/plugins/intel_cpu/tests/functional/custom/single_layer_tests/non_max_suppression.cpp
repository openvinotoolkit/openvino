// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/test_enums.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

enum {
    BATCHES,
    BOXES,
    CLASSES
};

using TargetShapeParams = std::tuple<size_t,   // Number of batches
                                     size_t,   // Number of boxes
                                     size_t>;  // Number of classes

using InputShapeParams = std::tuple<std::vector<ov::Dimension>,       // bounds for input dynamic shape
                                    std::vector<TargetShapeParams>>;  // target input dimensions

using InputPrecisions = std::tuple<ElementType,  // boxes and scores precisions
                                   ElementType,  // max_output_boxes_per_class precision
                                   ElementType>; // iou_threshold, score_threshold, soft_nms_sigma precisions

using ThresholdValues = std::tuple<float,  // IOU threshold
                                   float,  // Score threshold
                                   float>; // Soft NMS sigma

using NmsParams = std::tuple<InputShapeParams,                                   // Params using to create 1st and 2nd inputs
                             InputPrecisions,                                    // Input precisions
                             int32_t,                                            // Max output boxes per class
                             ThresholdValues,                                    // IOU, Score, Soft NMS sigma
                             ov::test::utils::InputLayerType,                    // max_output_boxes_per_class input type
                             ov::op::v9::NonMaxSuppression::BoxEncodingType,     // Box encoding
                             bool,                                               // Sort result descending
                             ElementType,                                        // Output type
                             std::string>;                                       // Device name

class NmsLayerCPUTest : public testing::WithParamInterface<NmsParams>, virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsParams>& obj) {
        InputShapeParams inShapeParams;
        InputPrecisions inPrecisions;
        int32_t maxOutBoxesPerClass;
        ov::test::utils::InputLayerType maxOutBoxesType;
        ThresholdValues thrValues;
        float iouThr, scoreThr, softNmsSigma;
        ov::op::v9::NonMaxSuppression::BoxEncodingType boxEncoding;
        bool sortResDescend;
        ElementType outType;
        std::string targetDevice;
        std::tie(inShapeParams, inPrecisions, maxOutBoxesPerClass, thrValues, maxOutBoxesType, boxEncoding, sortResDescend, outType, targetDevice) = obj.param;

        std::tie(iouThr, scoreThr, softNmsSigma) = thrValues;

        ElementType paramsPrec, maxBoxPrec, thrPrec;
        std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

        std::vector<ov::Dimension> bounds;
        std::vector<TargetShapeParams> targetShapes;
        std::tie(bounds, targetShapes) = inShapeParams;

        std::ostringstream result;
        if (!bounds.empty()) {
            OPENVINO_ASSERT(bounds.size() == 3);
            result << "BatchesBounds=" << bounds[BATCHES] << "_BoxesBounds=" << bounds[BOXES] << "_ClassesBounds=" << bounds[CLASSES] << "_";
        }
        for (const auto &ts : targetShapes) {
            size_t numBatches, numBoxes, numClasses;
            std::tie(numBatches, numBoxes, numClasses) = ts;
            result << "(nB=" << numBatches << "_nBox=" << numBoxes << "_nC=" << numClasses << ")_";
        }
        using ov::operator<<;
        result << "paramsPrec=" << paramsPrec << "_maxBoxPrec=" << maxBoxPrec << "_thrPrec=" << thrPrec << "_";
        result << "maxOutBoxesPerClass=" << maxOutBoxesPerClass << "_";
        result << "iouThr=" << iouThr << "_scoreThr=" << scoreThr << "_softNmsSigma=" << softNmsSigma << "_";
        result << "maxOutBoxesType=" << maxOutBoxesType << "_";
        result << "boxEncoding=" << boxEncoding << "_sortResDescend=" << sortResDescend << "_outType=" << outType << "_";
        result << "TargetDevice=" << targetDevice;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        SubgraphBaseTest::generate_inputs(targetInputStaticShapes);
        // w/a to fill valid data for port 2
        const auto& funcInputs = function->inputs();
        if (funcInputs.size() < 3) return;
        auto node = funcInputs[2].get_node_shared_ptr();
        auto it = inputs.find(node);
        if (it == inputs.end()) return;
        auto tensor = ov::Tensor(node->get_element_type(), targetInputStaticShapes[2], &maxOutBoxesPerClass);
        inputs[node] = tensor;
    }

    void compare(const std::vector<ov::Tensor> &expected, const std::vector<ov::Tensor> &actual) override {
        CompareBBoxes(expected, actual);
        inferRequestNum++;
    }

protected:
    void SetUp() override {
        InputShapeParams inShapeParams;
        InputPrecisions inPrecisions;
        ThresholdValues thrValues;
        ov::test::utils::InputLayerType maxOutBoxesType;
        float iouThr, scoreThr, softNmsSigma;
        ov::op::v9::NonMaxSuppression::BoxEncodingType boxEncoding;
        bool sortResDescend;
        ElementType outType;
        std::tie(inShapeParams, inPrecisions, maxOutBoxesPerClass, thrValues, maxOutBoxesType, boxEncoding, sortResDescend, outType,
                 targetDevice) = this->GetParam();
        ElementType paramsPrec, maxBoxPrec, thrPrec;
        std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

        std::tie(iouThr, scoreThr, softNmsSigma) = thrValues;

        std::vector<ov::Dimension> bounds;
        std::tie(bounds, targetInDims) = inShapeParams;

        if (!bounds.empty()) {
            inputDynamicShapes = std::vector<ov::PartialShape>{{bounds[BATCHES], bounds[BOXES], 4}, {bounds[BATCHES], bounds[CLASSES], bounds[BOXES]}};
        } else {
            size_t batches, boxes, classes;
            std::tie(batches, boxes, classes) = targetInDims.front();
            ov::Dimension numBatches(batches), numBoxes(boxes), numClasses(classes);
            inputDynamicShapes = std::vector<ov::PartialShape>{{numBatches, numBoxes, 4}, {numBatches, numClasses, numBoxes}};
        }

        for (const auto &ts : targetInDims) {
            size_t numBatches, numBoxes, numClasses;
            std::tie(numBatches, numBoxes, numClasses) = ts;
            targetStaticShapes.push_back(std::vector<ov::Shape>{{numBatches, numBoxes, 4}, {numBatches, numClasses, numBoxes}});
            if (maxOutBoxesType == ov::test::utils::InputLayerType::PARAMETER) {
                targetStaticShapes.back().push_back(ov::Shape{1});
            }
        }

        std::shared_ptr<ov::Node> maxOutBoxesPerClassNode;
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(paramsPrec, shape));
        }
        params[0]->set_friendly_name("param_1");
        params[1]->set_friendly_name("param_2");

        if (maxOutBoxesType == ov::test::utils::InputLayerType::PARAMETER) {
            inputDynamicShapes.push_back(ov::PartialShape{1});
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ElementType::i32, inputDynamicShapes.back()));
            params[1]->set_friendly_name("param_3");
            maxOutBoxesPerClassNode = params.back();
        } else {
            maxOutBoxesPerClassNode = ov::test::utils::deprecated::make_constant(maxBoxPrec, ov::Shape{}, std::vector<int32_t>{maxOutBoxesPerClass});
        }

        auto iouThrNode = ov::test::utils::deprecated::make_constant(thrPrec, ov::Shape{}, std::vector<float>{iouThr})->output(0);
        auto scoreThrNode = ov::test::utils::deprecated::make_constant(thrPrec, ov::Shape{}, std::vector<float>{scoreThr})->output(0);
        auto softNmsSigmaNode = ov::test::utils::deprecated::make_constant(thrPrec, ov::Shape{}, std::vector<float>{softNmsSigma})->output(0);
        auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(params[0], params[1], maxOutBoxesPerClassNode, iouThrNode, scoreThrNode,
                                                                       softNmsSigmaNode, boxEncoding, sortResDescend, outType);

        function = makeNgraphFunction(paramsPrec, params, nms, "NMS");
    }

private:
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
    void CompareBBoxes(const std::vector<ov::Tensor> &expectedOutputs, const std::vector<ov::Tensor> &actualOutputs) {
        size_t numBatches, numBoxes, numClasses;
        std::tie(numBatches, numBoxes, numClasses) = targetInDims[inferRequestNum];

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
            std::pair<std::shared_ptr<ov::Node>, ov::Tensor> bboxes = *inputs.begin();
            for (const auto &input : inputs) {
                if (input.first->get_name() < bboxes.first->get_name()) {
                    bboxes = input;
                }
            }

            const auto buffer = bboxes.second.data<float>();
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
            const auto indeces_iter = expectedOutputs.begin();
            const auto scores_iter = expectedOutputs.begin() + 1;
            size_t selected_indices_size = indeces_iter->get_size();
            size_t selected_scores_size = scores_iter->get_size();
            ASSERT_TRUE(selected_indices_size == selected_scores_size);

            expectedList.resize(selected_indices_size);

            if (indeces_iter->get_element_type() == ElementType::i32) {
                auto selected_indices_data = indeces_iter->data<int32_t>();

                for (size_t i = 0; i < selected_indices_size; i += 3) {
                    expectedList[i/3].batchId = selected_indices_data[i+0];
                    expectedList[i/3].classId = selected_indices_data[i+1];
                    expectedList[i/3].boxId   = selected_indices_data[i+2];
                    expectedList[i/3].rect    = coordList[expectedList[i/3].batchId][expectedList[i/3].boxId];
                }
            } else {
                auto selected_indices_data = indeces_iter->data<int64_t>();

                for (size_t i = 0; i < selected_indices_size; i += 3) {
                    expectedList[i/3].batchId = static_cast<int32_t>(selected_indices_data[i+0]);
                    expectedList[i/3].classId = static_cast<int32_t>(selected_indices_data[i+1]);
                    expectedList[i/3].boxId   = static_cast<int32_t>(selected_indices_data[i+2]);
                    expectedList[i/3].rect    = coordList[expectedList[i/3].batchId][expectedList[i/3].boxId];
                }
            }

            if (scores_iter->get_element_type() == ElementType::f32) {
                auto selected_scores_data = scores_iter->data<float>();
                for (size_t i = 0; i < selected_scores_size; i += 3) {
                    expectedList[i/3].score = selected_scores_data[i+2];
                }
            } else {
                auto selected_scores_data = scores_iter->data<double>();
                for (size_t i = 0; i < selected_scores_size; i += 3) {
                    expectedList[i/3].score = static_cast<float>(selected_scores_data[i+2]);
                }
            }

            std::sort(expectedList.begin(), expectedList.end(), compareBox);
        }

        // Get actual bboxes' index/score
        std::vector<Box> actualList;
        {
            const auto indeces_iter = actualOutputs.begin();
            const auto scores_iter = actualOutputs.begin() + 1;
            size_t selected_indices_size = indeces_iter->get_size();
            const auto selected_scores_data = scores_iter->data<float>();

            if (indeces_iter->get_element_type() == ElementType::i32) {
                const auto selected_indices_data = indeces_iter->data<int32_t>();
                for (size_t i = 0; i < selected_indices_size; i += 3) {
                    const int32_t batchId = selected_indices_data[i+0];
                    const int32_t classId = selected_indices_data[i+1];
                    const int32_t boxId   = selected_indices_data[i+2];
                    const float score = selected_scores_data[i+2];
                    if (batchId == -1 || classId == -1 || boxId == -1)
                        break;

                    actualList.emplace_back(batchId, classId, boxId, coordList[batchId][boxId], score);
                }
            } else {
                const auto selected_indices_data = indeces_iter->data<int64_t>();
                for (size_t i = 0; i < selected_indices_size; i += 3) {
                    const int32_t batchId = selected_indices_data[i+0];
                    const int32_t classId = selected_indices_data[i+1];
                    const int32_t boxId   = selected_indices_data[i+2];
                    const float score = selected_scores_data[i+2];
                    if (batchId == -1 || classId == -1 || boxId == -1)
                        break;

                    actualList.emplace_back(batchId, classId, boxId, coordList[batchId][boxId], score);
                }
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
    std::vector<TargetShapeParams> targetInDims;
    size_t inferRequestNum = 0;
    int32_t maxOutBoxesPerClass;
};

TEST_P(NmsLayerCPUTest, CompareWithRefs) {
    run();
    // CheckPluginRelatedResults(compiledModel, "NonMaxSuppression");
};

const std::vector<InputShapeParams> inShapeParams = {
    InputShapeParams{std::vector<ov::Dimension>{-1, -1, -1}, std::vector<TargetShapeParams>{TargetShapeParams{2, 50, 50},
                                                                                            TargetShapeParams{3, 100, 5},
                                                                                            TargetShapeParams{1, 10, 50}}},
    InputShapeParams{std::vector<ov::Dimension>{{1, 5}, {1, 100}, {10, 75}}, std::vector<TargetShapeParams>{TargetShapeParams{4, 15, 10},
                                                                                                            TargetShapeParams{5, 5, 12},
                                                                                                            TargetShapeParams{1, 35, 15}}}
};

const std::vector<int32_t> maxOutBoxPerClass = {5, 20};
const std::vector<float> threshold = {0.3f, 0.7f};
const std::vector<float> sigmaThreshold = {0.0f, 0.5f};
const std::vector<ov::op::v9::NonMaxSuppression::BoxEncodingType> encodType = {ov::op::v9::NonMaxSuppression::BoxEncodingType::CENTER,
                                                                               ov::op::v9::NonMaxSuppression::BoxEncodingType::CORNER};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<ElementType> outType = {ElementType::i32, ElementType::i64};
const std::vector<ov::test::utils::InputLayerType> maxBoxInputTypes = {ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT};

const auto nmsParams = ::testing::Combine(::testing::ValuesIn(inShapeParams),
                                          ::testing::Combine(::testing::Values(ElementType::f32),
                                                             ::testing::Values(ElementType::i32),
                                                             ::testing::Values(ElementType::f32)),
                                          ::testing::ValuesIn(maxOutBoxPerClass),
                                          ::testing::Combine(::testing::ValuesIn(threshold),
                                                             ::testing::ValuesIn(threshold),
                                                             ::testing::ValuesIn(sigmaThreshold)),
                                          ::testing::ValuesIn(maxBoxInputTypes),
                                          ::testing::ValuesIn(encodType),
                                          ::testing::ValuesIn(sortResDesc),
                                          ::testing::ValuesIn(outType),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_NmsLayerCPUTest, NmsLayerCPUTest, nmsParams, NmsLayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
