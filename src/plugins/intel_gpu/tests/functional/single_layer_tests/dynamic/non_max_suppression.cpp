// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/non_max_suppression.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/non_max_suppression.hpp"

namespace {

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

using InputPrecisions = std::tuple<ov::element::Type,  // boxes and scores precisions
                                   ov::element::Type,  // max_output_boxes_per_class precision
                                   ov::element::Type>; // iou_threshold, score_threshold, soft_nms_sigma precisions

using ThresholdValues = std::tuple<float,  // IOU threshold
                                   float,  // Score threshold
                                   float>; // Soft NMS sigma

using NmsLayerTestParams = std::tuple<InputShapeParams,                                   // Params using to create 1st and 2nd inputs
                                      InputPrecisions,                                    // Input precisions
                                      int32_t,                                            // Max output boxes per class
                                      ThresholdValues,                                    // IOU, Score, Soft NMS sigma
                                      ov::op::v9::NonMaxSuppression::BoxEncodingType,     // Box encoding
                                      bool,                                               // Sort result descending
                                      ov::element::Type,                              // Output type
                                      std::string,                                        // Device name
                                      std::map<std::string, std::string>>;                // Additional network configuration

class NmsLayerGPUTest : public testing::WithParamInterface<NmsLayerTestParams>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsLayerTestParams>& obj) {
        InputShapeParams inShapeParams;
        InputPrecisions inPrecisions;
        int32_t maxOutBoxesPerClass;
        ThresholdValues thrValues;
        float iouThr, scoreThr, softNmsSigma;
        ov::op::v9::NonMaxSuppression::BoxEncodingType boxEncoding;
        bool sortResDescend;
        ov::element::Type outType;
        std::string targetDevice;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inShapeParams, inPrecisions, maxOutBoxesPerClass, thrValues, boxEncoding, sortResDescend, outType,
                 targetDevice, additionalConfig) = obj.param;

        std::tie(iouThr, scoreThr, softNmsSigma) = thrValues;

        ov::element::Type paramsPrec, maxBoxPrec, thrPrec;
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
        result << "paramsPrec=" << paramsPrec << "_maxBoxPrec=" << maxBoxPrec << "_thrPrec=" << thrPrec << "_";
        result << "maxOutBoxesPerClass=" << maxOutBoxesPerClass << "_";
        result << "iouThr=" << iouThr << "_scoreThr=" << scoreThr << "_softNmsSigma=" << softNmsSigma << "_";
        using ov::operator<<;
        result << "boxEncoding=" << boxEncoding << "_sortResDescend=" << sortResDescend << "_outType=" << outType << "_";
        result << "config=(";
        for (const auto& configEntry : additionalConfig) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")_";
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
        std::pair<std::shared_ptr<ov::Node>, ov::Tensor> bboxes = *inputs.begin();
        for (const auto &input : inputs) {
            if (input.first->get_name() < bboxes.first->get_name()) {
                bboxes = input;
            }
        }
        size_t numBatches, numBoxes, numClasses;
        std::tie(numBatches, numBoxes, numClasses) = targetInDims[inferRequestNum];
        ov::test::compare_b_boxes(expected, actual, bboxes.second, numBatches, numBoxes);
        inferRequestNum++;
    }

protected:
    void SetUp() override {
        InputShapeParams inShapeParams;
        InputPrecisions inPrecisions;
        ThresholdValues thrValues;
        float iouThr, scoreThr, softNmsSigma;
        ov::op::v9::NonMaxSuppression::BoxEncodingType boxEncoding;
        bool sortResDescend;
        ov::element::Type outType;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inShapeParams, inPrecisions, maxOutBoxesPerClass, thrValues, boxEncoding, sortResDescend, outType,
                 targetDevice, additionalConfig) = this->GetParam();
        ov::element::Type paramsPrec, maxBoxPrec, thrPrec;
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
        }

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(paramsPrec, shape));
        }
        params[0]->set_friendly_name("param_1");
        params[1]->set_friendly_name("param_2");

        auto maxOutBoxesPerClassNode = std::make_shared<ov::op::v0::Constant>(maxBoxPrec, ov::Shape{}, std::vector<int32_t>{maxOutBoxesPerClass});
        auto iouThrNode = std::make_shared<ov::op::v0::Constant>(thrPrec, ov::Shape{}, std::vector<float>{iouThr});
        auto scoreThrNode = std::make_shared<ov::op::v0::Constant>(thrPrec, ov::Shape{}, std::vector<float>{scoreThr});
        auto softNmsSigmaNode = std::make_shared<ov::op::v0::Constant>(thrPrec, ov::Shape{}, std::vector<float>{softNmsSigma});

        auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(params[0], params[1], maxOutBoxesPerClassNode, iouThrNode, scoreThrNode,
                                                                   softNmsSigmaNode, boxEncoding, sortResDescend, outType);
        ov::ResultVector results;
        for (size_t i = 0; i < nms->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(nms->output(i)));
        }
        function = std::make_shared<ov::Model>(results, params, "Nms");
    }

private:
    std::vector<TargetShapeParams> targetInDims;
    size_t inferRequestNum = 0;
    int32_t maxOutBoxesPerClass;
};

TEST_P(NmsLayerGPUTest, Inference) {
    run();
}

std::map<std::string, std::string> emptyAdditionalConfig;

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
const std::vector<ov::op::v9::NonMaxSuppression::BoxEncodingType> encodType =
    {ov::op::v9::NonMaxSuppression::BoxEncodingType::CENTER,
     ov::op::v9::NonMaxSuppression::BoxEncodingType::CORNER};

const std::vector<bool> sortResDesc = {true, false};
const std::vector<ov::element::Type> outType = {ov::element::i32};

INSTANTIATE_TEST_SUITE_P(smoke_Nms_dynamic, NmsLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapeParams),
        ::testing::Combine(
            ::testing::Values(ov::element::f32),
            ::testing::Values(ov::element::i32),
            ::testing::Values(ov::element::f32)),
        ::testing::ValuesIn(maxOutBoxPerClass),
        ::testing::Combine(
            ::testing::ValuesIn(threshold),
            ::testing::ValuesIn(threshold),
            ::testing::ValuesIn(sigmaThreshold)),
        ::testing::ValuesIn(encodType),
        ::testing::ValuesIn(sortResDesc),
        ::testing::ValuesIn(outType),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(emptyAdditionalConfig)),
    NmsLayerGPUTest::getTestCaseName);

} // namespace
