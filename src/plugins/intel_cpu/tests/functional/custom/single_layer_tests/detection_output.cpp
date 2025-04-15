// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace DetectionOutput {
static std::ostream& operator<<(std::ostream& result, const ov::op::v0::DetectionOutput::Attributes& attrs) {
    result << "Classes=" << attrs.num_classes << "_";
    result << "backgrId=" << attrs.background_label_id << "_";
    result << "topK="  << attrs.top_k << "_";
    result << "varEnc=" << attrs.variance_encoded_in_target << "_";
    result << "keepTopK=" << ov::test::utils::vec2str(attrs.keep_top_k) << "_";
    result << "codeType=" << attrs.code_type << "_";
    result << "shareLoc=" << attrs.share_location << "_";
    result << "nmsThr=" << attrs.nms_threshold << "_";
    result << "confThr=" << attrs.confidence_threshold << "_";
    result << "clipAfterNms=" << attrs.clip_after_nms << "_";
    result << "clipBeforeNms=" << attrs.clip_before_nms << "_";
    result << "decrId=" << attrs.decrease_label_id << "_";
    result << "norm=" << attrs.normalized << "_";
    result << "inH=" << attrs.input_height << "_";
    result << "inW=" << attrs.input_width << "_";
    result << "OS=" << attrs.objectness_score << "_";
    return result;
}
}  // namespace DetectionOutput

using namespace CPUTestUtils;
namespace ov {
namespace test {

enum { idxLocation, idxConfidence, idxPriors, idxArmConfidence, idxArmLocation, numInputs };

using ParamsWhichSizeDependsDynamic = std::tuple<bool,                  // varianceEncodedInTarget
                                                 bool,                  // shareLocation
                                                 bool,                  // normalized
                                                 size_t,                // inputHeight
                                                 size_t,                // inputWidth
                                                 ov::test::InputShape,  // "Location" input
                                                 ov::test::InputShape,  // "Confidence" input
                                                 ov::test::InputShape,  // "Priors" input
                                                 ov::test::InputShape,  // "ArmConfidence" input
                                                 ov::test::InputShape   // "ArmLocation" input
                                                 >;

using DetectionOutputAttributes = std::tuple<int,               // numClasses
                                             int,               // backgroundLabelId
                                             int,               // topK
                                             std::vector<int>,  // keepTopK
                                             std::string,       // codeType
                                             float,             // nmsThreshold
                                             float,             // confidenceThreshold
                                             bool,              // clip_afterNms
                                             bool,              // clip_beforeNms
                                             bool               // decreaseLabelId
                                             >;

using DetectionOutputParamsDynamic = std::tuple<DetectionOutputAttributes,
                                                ParamsWhichSizeDependsDynamic,
                                                size_t,      // Number of batch
                                                float,       // objectnessScore
                                                bool,        // replace dynamic shapes to intervals
                                                std::string  // Device name
                                                >;

class DetectionOutputLayerCPUTest : public testing::WithParamInterface<DetectionOutputParamsDynamic>,
                                    virtual public SubgraphBaseTest,
                                    public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DetectionOutputParamsDynamic>& obj) {
        DetectionOutputAttributes commonAttrs;
        ParamsWhichSizeDependsDynamic specificAttrs;
        ov::op::v0::DetectionOutput::Attributes attrs;
        size_t batch;
        bool replaceDynamicShapesToIntervals;
        std::string targetDevice;
        std::tie(commonAttrs,
                 specificAttrs,
                 batch,
                 attrs.objectness_score,
                 replaceDynamicShapesToIntervals,
                 targetDevice) = obj.param;

        std::tie(attrs.num_classes,
                 attrs.background_label_id,
                 attrs.top_k,
                 attrs.keep_top_k,
                 attrs.code_type,
                 attrs.nms_threshold,
                 attrs.confidence_threshold,
                 attrs.clip_after_nms,
                 attrs.clip_before_nms,
                 attrs.decrease_label_id) = commonAttrs;

        const size_t numInputs = 5;
        std::vector<ov::test::InputShape> inShapes(numInputs);
        std::tie(attrs.variance_encoded_in_target,
                 attrs.share_location,
                 attrs.normalized,
                 attrs.input_height,
                 attrs.input_width,
                 inShapes[idxLocation],
                 inShapes[idxConfidence],
                 inShapes[idxPriors],
                 inShapes[idxArmConfidence],
                 inShapes[idxArmLocation]) = specificAttrs;

        if (inShapes[idxArmConfidence].first.rank().get_length() == 0ul) {
            inShapes.resize(3);
        }

        for (size_t i = 0; i < inShapes.size(); i++) {
            inShapes[i].first[0] = batch;
        }

        std::ostringstream result;
        result << "IS = { ";

        using ov::test::operator<<;
        result << "LOC=" << inShapes[0] << "_";
        result << "CONF=" << inShapes[1] << "_";
        result << "PRIOR=" << inShapes[2];
        if (inShapes.size() > 3) {
            result << "_ARM_CONF=" << inShapes[3] << "_";
            result << "ARM_LOC=" << inShapes[4] << " }_";
        }

        using DetectionOutput::operator<<;
        result << attrs;
        result << "RDS=" << (replaceDynamicShapesToIntervals ? "true" : "false") << "_";
        result << "TargetDevice=" << targetDevice;
        return result.str();
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(expectedTensors.size(), actualTensors.size());

        for (size_t i = 0; i < expectedTensors.size(); ++i) {
            auto expected = expectedTensors[i];
            auto actual = actualTensors[i];
            ASSERT_EQ(expected.get_size(), actual.get_size());

            size_t expSize = 0;
            const float* expBuf = expected.data<const float>();
            for (size_t i = 0; i < expected.get_size(); i += 7) {
                if (expBuf[i] == -1)
                    break;
                expSize += 7;
            }

            size_t actSize = 0;
            const float* actBuf = actual.data<const float>();
            for (size_t i = 0; i < actual.get_size(); i += 7) {
                if (actBuf[i] == -1)
                    break;
                actSize += 7;
            }

            ASSERT_EQ(expSize, actSize);
        }

        ov::test::SubgraphBaseTest::compare(expectedTensors, actualTensors);
    }

    void SetUp() override {
        DetectionOutputAttributes commonAttrs;
        ParamsWhichSizeDependsDynamic specificAttrs;
        size_t batch;
        bool replaceDynamicShapesToIntervals;
        std::tie(commonAttrs,
                 specificAttrs,
                 batch,
                 attrs.objectness_score,
                 replaceDynamicShapesToIntervals,
                 targetDevice) = this->GetParam();

        std::tie(attrs.num_classes,
                 attrs.background_label_id,
                 attrs.top_k,
                 attrs.keep_top_k,
                 attrs.code_type,
                 attrs.nms_threshold,
                 attrs.confidence_threshold,
                 attrs.clip_after_nms,
                 attrs.clip_before_nms,
                 attrs.decrease_label_id) = commonAttrs;

        inShapes.resize(numInputs);
        std::tie(attrs.variance_encoded_in_target,
                 attrs.share_location,
                 attrs.normalized,
                 attrs.input_height,
                 attrs.input_width,
                 inShapes[idxLocation],
                 inShapes[idxConfidence],
                 inShapes[idxPriors],
                 inShapes[idxArmConfidence],
                 inShapes[idxArmLocation]) = specificAttrs;

        if (inShapes[idxArmConfidence].first.rank().get_length() == 0) {
            inShapes.resize(3);
        }

        if (replaceDynamicShapesToIntervals) {
            set_dimension_intervals(inShapes);
        }

        for (auto& value : inShapes) {
            auto shapes = value.second;
            for (auto& shape : shapes) {
                shape[0] = batch;
            }
        }

        init_input_shapes({inShapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
            params.push_back(param);
        }
        std::shared_ptr<ov::Node> detOut;
        if (params.size() == 3)
            detOut = std::make_shared<ov::op::v0::DetectionOutput>(params[0], params[1], params[2], attrs);
        else if (params.size() == 5)
            detOut = std::make_shared<ov::op::v0::DetectionOutput>(params[0],
                                                                   params[1],
                                                                   params[2],
                                                                   params[3],
                                                                   params[4],
                                                                   attrs);
        else
            OPENVINO_THROW("DetectionOutput layer supports only 3 or 5 inputs");

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(detOut)};
        function = std::make_shared<ov::Model>(results, params, "DetectionOutputDynamic");
    }

private:
    // define dynamic shapes dimension intervals
    static void set_dimension_intervals(std::vector<std::pair<ov::PartialShape, std::vector<ov::Shape>>>& inputShapes) {
        for (auto& input_shape : inputShapes) {
            const auto model_dynamic_shape = input_shape.first;
            OPENVINO_ASSERT(model_dynamic_shape.is_dynamic(), "input shape is not dynamic");

            const auto inputShapeRank = model_dynamic_shape.rank();
            OPENVINO_ASSERT(!inputShapeRank.is_dynamic(), "input shape rank is dynamic");

            for (auto dimension = 0; dimension < inputShapeRank.get_length(); ++dimension) {
                auto interval_min = -1;
                auto interval_max = 0;
                for (auto& input_static_shape : input_shape.second) {
                    if ((interval_min == -1) || (static_cast<size_t>(interval_min) > input_static_shape[dimension])) {
                        interval_min = input_static_shape[dimension];
                    }
                    if (static_cast<size_t>(interval_max) < input_static_shape[dimension]) {
                        interval_max = input_static_shape[dimension];
                    }
                }

                input_shape.first[dimension] = {interval_min,
                                                interval_min == interval_max ? (interval_max + 1) : interval_max};
            }
        }
    }
    ov::op::v0::DetectionOutput::Attributes attrs;
    std::vector<ov::test::InputShape> inShapes;
};

TEST_P(DetectionOutputLayerCPUTest, CompareWithRefs) {
    run();
}

namespace {

const int numClasses = 11;
const int backgroundLabelId = 0;
const std::vector<int> topK = {75};
const std::vector<std::vector<int>> keepTopK = {{50}, {100}};
const std::vector<std::string> codeType = {"caffe.PriorBoxParameter.CORNER", "caffe.PriorBoxParameter.CENTER_SIZE"};
const float nmsThreshold = 0.5f;
const float confidenceThreshold = 0.3f;
const std::vector<bool> clipAfterNms = {true, false};
const std::vector<bool> clipBeforeNms = {true, false};
const std::vector<bool> decreaseLabelId = {true, false};
const float objectnessScore = 0.4f;
const std::vector<size_t> numberBatch = {1, 2};

const auto commonAttributes = ::testing::Combine(::testing::Values(numClasses),
                                                 ::testing::Values(backgroundLabelId),
                                                 ::testing::ValuesIn(topK),
                                                 ::testing::ValuesIn(keepTopK),
                                                 ::testing::ValuesIn(codeType),
                                                 ::testing::Values(nmsThreshold),
                                                 ::testing::Values(confidenceThreshold),
                                                 ::testing::ValuesIn(clipAfterNms),
                                                 ::testing::ValuesIn(clipBeforeNms),
                                                 ::testing::ValuesIn(decreaseLabelId));

/* =============== 3 inputs cases =============== */

const std::vector<ParamsWhichSizeDependsDynamic> specificParams3InDynamic = {
    // dynamic input shapes
    ParamsWhichSizeDependsDynamic{true,
                                  true,
                                  true,
                                  1,
                                  1,
                                  {// input model dynamic shapes
                                   {ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                   // input tensor shapes
                                   {{1, 60}, {1, 120}}},
                                  {// input model dynamic shapes
                                   {ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                   // input tensor shapes
                                   {{1, 165}, {1, 330}}},
                                  {// input model dynamic shapes
                                   {ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                   // input tensor shapes
                                   {{1, 1, 60}, {1, 1, 120}}},
                                  {},
                                  {}},
    ParamsWhichSizeDependsDynamic{
        true,
        false,
        true,
        1,
        1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 60}, {1, 1, 120}}},
        {},
        {}},
    ParamsWhichSizeDependsDynamic{
        false,
        true,
        true,
        1,
        1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {},
        {}},
    ParamsWhichSizeDependsDynamic{
        false,
        false,
        true,
        1,
        1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {},
        {}},
    ParamsWhichSizeDependsDynamic{
        true,
        true,
        false,
        10,
        10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {},
        {}},
    ParamsWhichSizeDependsDynamic{
        true,
        false,
        false,
        10,
        10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {},
        {}},
    ParamsWhichSizeDependsDynamic{
        false,
        true,
        false,
        10,
        10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {},
        {}},
    ParamsWhichSizeDependsDynamic{
        false,
        false,
        false,
        10,
        10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {},
        {}},
};

const auto params3InputsDynamic = ::testing::Combine(commonAttributes,
                                                     ::testing::ValuesIn(specificParams3InDynamic),
                                                     ::testing::ValuesIn(numberBatch),
                                                     ::testing::Values(0.0f),
                                                     ::testing::Values(false, true),
                                                     ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_CPUDetectionOutputDynamic3In,
                         DetectionOutputLayerCPUTest,
                         params3InputsDynamic,
                         DetectionOutputLayerCPUTest::getTestCaseName);

//////////////////large tensor/////////////////
// There are two major implemenation for DO node, sparsity and dense manner.
// This test config(shapes, threshold...) go to sparsity path in most machines(as long as L3 per core cache is smaller
// than 8M).
const std::vector<ParamsWhichSizeDependsDynamic> specificParams3InDynamicLargeTensor = {
    // dynamic input shapes
    ParamsWhichSizeDependsDynamic{true,
                                  true,
                                  true,
                                  1,
                                  1,
                                  {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 381360}, {1, 381360}}},
                                  {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1048740}, {1, 1048740}}},
                                  {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                   {{1, 1, 381360}, {1, 1, 381360}}},
                                  {},
                                  {}},
    ParamsWhichSizeDependsDynamic{false,
                                  true,
                                  true,
                                  1,
                                  1,
                                  {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 381360}, {1, 381360}}},
                                  {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1048740}, {1, 1048740}}},
                                  {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                   {{1, 1, 381360}, {1, 1, 381360}}},
                                  {},
                                  {}},
};

const std::vector<float> confThreshold = {0.032f, 0.88f};
const auto commonAttributesLargeTensor = ::testing::Combine(::testing::Values(numClasses),
                                                            ::testing::Values(backgroundLabelId),
                                                            ::testing::ValuesIn(topK),
                                                            ::testing::ValuesIn(keepTopK),
                                                            ::testing::ValuesIn(codeType),
                                                            ::testing::Values(nmsThreshold),
                                                            ::testing::ValuesIn(confThreshold),
                                                            ::testing::ValuesIn(clipAfterNms),
                                                            ::testing::ValuesIn(clipBeforeNms),
                                                            ::testing::Values(false));

const auto params3InputsDynamicLargeTensor =
    ::testing::Combine(commonAttributesLargeTensor,
                       ::testing::ValuesIn(specificParams3InDynamicLargeTensor),
                       ::testing::ValuesIn(numberBatch),
                       ::testing::Values(0.0f),
                       ::testing::Values(false, true),
                       ::testing::Values(ov::test::utils::DEVICE_CPU));
INSTANTIATE_TEST_SUITE_P(CPUDetectionOutputDynamic3InLargeTensor,
                         DetectionOutputLayerCPUTest,
                         params3InputsDynamicLargeTensor,
                         DetectionOutputLayerCPUTest::getTestCaseName);

/* =============== 5 inputs cases =============== */

const std::vector<ParamsWhichSizeDependsDynamic> specificParams5InDynamic = {
    // dynamic input shapes
    ParamsWhichSizeDependsDynamic{
        true,
        true,
        true,
        1,
        1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 60}, {1, 1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
    },
    ParamsWhichSizeDependsDynamic{
        true,
        false,
        true,
        1,
        1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 60}, {1, 1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
    },
    ParamsWhichSizeDependsDynamic{
        false,
        true,
        true,
        1,
        1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}}},
    ParamsWhichSizeDependsDynamic{
        false,
        false,
        true,
        1,
        1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}}},

    ParamsWhichSizeDependsDynamic{
        true,
        true,
        false,
        10,
        10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}}},
    ParamsWhichSizeDependsDynamic{
        true,
        false,
        false,
        10,
        10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}}},
    ParamsWhichSizeDependsDynamic{
        false,
        true,
        false,
        10,
        10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}}},
    ParamsWhichSizeDependsDynamic{
        false,
        false,
        false,
        10,
        10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}}},
};

const auto params5InputsDynamic = ::testing::Combine(commonAttributes,
                                                     ::testing::ValuesIn(specificParams5InDynamic),
                                                     ::testing::ValuesIn(numberBatch),
                                                     ::testing::Values(objectnessScore),
                                                     ::testing::Values(false, true),
                                                     ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_CPUDetectionOutputDynamic5In,
                         DetectionOutputLayerCPUTest,
                         params5InputsDynamic,
                         DetectionOutputLayerCPUTest::getTestCaseName);

//////////////////large tensor/////////////////
const std::vector<ParamsWhichSizeDependsDynamic> specificParams5InDynamicLargeTensor = {
    // dynamic input shapes
    ParamsWhichSizeDependsDynamic{
        true,
        true,
        true,
        1,
        1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 381360}, {1, 381360}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1048740}, {1, 1048740}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
         {{1, 1, 381360}, {1, 1, 381360}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 190680}, {1, 190680}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 381360}, {1, 381360}}},
    },
    ParamsWhichSizeDependsDynamic{
        true,
        false,
        true,
        1,
        1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 4194960}, {1, 4194960}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1048740}, {1, 1048740}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
         {{1, 1, 381360}, {1, 1, 381360}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 190680}, {1, 190680}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 4194960}, {1, 4194960}}},
    },
};
const auto params5InputsDynamicLargeTensor =
    ::testing::Combine(commonAttributesLargeTensor,
                       ::testing::ValuesIn(specificParams5InDynamicLargeTensor),
                       ::testing::ValuesIn(numberBatch),
                       ::testing::Values(objectnessScore),
                       ::testing::Values(false, true),
                       ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(CPUDetectionOutputDynamic5InLargeTensor,
                         DetectionOutputLayerCPUTest,
                         params5InputsDynamicLargeTensor,
                         DetectionOutputLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
