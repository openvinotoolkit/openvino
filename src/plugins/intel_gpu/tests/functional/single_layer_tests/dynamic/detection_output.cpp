// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/detection_output.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include <string>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

enum {
    idxLocation,
    idxConfidence,
    idxPriors,
    idxArmConfidence,
    idxArmLocation,
    numInputs
};

typedef std::tuple<
    bool,                 // varianceEncodedInTarget
    bool,                 // shareLocation
    bool,                 // normalized
    size_t,               // inputHeight
    size_t,               // inputWidth
    ov::test::InputShape, // "Location" input
    ov::test::InputShape, // "Confidence" input
    ov::test::InputShape, // "Priors" input
    ov::test::InputShape, // "ArmConfidence" input
    ov::test::InputShape  // "ArmLocation" input
> ParamsWhichSizeDependsDynamic;

typedef std::tuple<
    int,                // numClasses
    int,                // backgroundLabelId
    int,                // topK
    std::vector<int>,   // keepTopK
    std::string,        // codeType
    float,              // nmsThreshold
    float,              // confidenceThreshold
    bool,               // clip_afterNms
    bool,               // clip_beforeNms
    bool                // decreaseLabelId
> DetectionOutputAttributes;

typedef std::tuple<
    DetectionOutputAttributes,
    ParamsWhichSizeDependsDynamic,
    size_t,     // Number of batch
    float,      // objectnessScore
    bool,       // replace dynamic shapes to intervals
    std::string // Device name
> DetectionOutputGPUTestParams;

class DetectionOutputLayerGPUTest : public testing::WithParamInterface<DetectionOutputGPUTestParams>,
                                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DetectionOutputGPUTestParams> obj) {
        DetectionOutputAttributes commonAttrs;
        ParamsWhichSizeDependsDynamic specificAttrs;
        ngraph::op::DetectionOutputAttrs attrs;
        size_t batch;
        bool replaceDynamicShapesToIntervals;
        std::string targetDevice;
        std::tie(commonAttrs, specificAttrs, batch, attrs.objectness_score, replaceDynamicShapesToIntervals, targetDevice) = obj.param;

        std::tie(attrs.num_classes, attrs.background_label_id, attrs.top_k, attrs.keep_top_k, attrs.code_type, attrs.nms_threshold, attrs.confidence_threshold,
                 attrs.clip_after_nms, attrs.clip_before_nms, attrs.decrease_label_id) = commonAttrs;

        const size_t numInputs = 5;
        std::vector<ov::test::InputShape> inShapes(numInputs);
        std::tie(attrs.variance_encoded_in_target, attrs.share_location, attrs.normalized, attrs.input_height, attrs.input_width,
                 inShapes[idxLocation], inShapes[idxConfidence], inShapes[idxPriors], inShapes[idxArmConfidence], inShapes[idxArmLocation]) = specificAttrs;

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
        } else {
            result << " }_";
        }

        using LayerTestsDefinitions::operator<<;
        result << attrs;
        result << "RDS=" << (replaceDynamicShapesToIntervals ? "true" : "false") << "_";
        result << "TargetDevice=" << targetDevice;
        return result.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (auto i = 0ul; i < funcInputs.size(); ++i) {
            const auto &funcInput = funcInputs[i];
            InferenceEngine::Blob::Ptr blob;
            int32_t resolution = 1;
            uint32_t range = 1;
            if (i == 2) {
                if (attrs.normalized) {
                    resolution = 100;
                } else {
                    range = 10;
                }
            } else if (i == 1 || i == 3) {
                resolution = 1000;
            } else {
                resolution = 10;
            }

            auto tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], range, 0, resolution);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors, const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(expectedTensors.size(), actualTensors.size());

        for (size_t i = 0; i < expectedTensors.size(); ++i) {
            auto expected = expectedTensors[i];
            auto actual = actualTensors[i];
            ASSERT_EQ(expected.get_size(), actual.get_size());

            size_t expSize = 0;
            const float* expBuf = expected.data<const float>();
            for (size_t i = 0; i < expected.get_size(); i+=7) {
                if (expBuf[i] == -1)
                    break;
                expSize += 7;
            }

            size_t actSize = 0;
            const float* actBuf = actual.data<const float>();
            for (size_t i = 0; i < actual.get_size(); i+=7) {
                if (actBuf[i] == -1)
                    break;
                actSize += 7;
            }

            ASSERT_EQ(expSize, actSize);
        }

        ov::test::SubgraphBaseTest::compare(expectedTensors, actualTensors);
    }

protected:
    void SetUp() override {
        DetectionOutputAttributes commonAttrs;
        ParamsWhichSizeDependsDynamic specificAttrs;
        size_t batch;
        bool replaceDynamicShapesToIntervals;
        std::tie(commonAttrs, specificAttrs, batch, attrs.objectness_score, replaceDynamicShapesToIntervals, targetDevice) = this->GetParam();

        std::tie(attrs.num_classes, attrs.background_label_id, attrs.top_k, attrs.keep_top_k, attrs.code_type, attrs.nms_threshold, attrs.confidence_threshold,
                 attrs.clip_after_nms, attrs.clip_before_nms, attrs.decrease_label_id) = commonAttrs;

        inShapes.resize(numInputs);
        std::tie(attrs.variance_encoded_in_target, attrs.share_location, attrs.normalized, attrs.input_height, attrs.input_width,
                 inShapes[idxLocation], inShapes[idxConfidence], inShapes[idxPriors], inShapes[idxArmConfidence], inShapes[idxArmLocation]) = specificAttrs;

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

        init_input_shapes({ inShapes });

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, shape));
        }
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));

        if (attrs.num_classes == -1) {
            std::shared_ptr<ov::op::v8::DetectionOutput> detOut;

            if (paramOuts.size() == 3)
                detOut = std::make_shared<ov::op::v8::DetectionOutput>(paramOuts[0], paramOuts[1], paramOuts[2], attrs);
            else if (paramOuts.size() == 5)
                detOut = std::make_shared<ov::op::v8::DetectionOutput>(paramOuts[0], paramOuts[1], paramOuts[2], paramOuts[3], paramOuts[4], attrs);
            else
                throw std::runtime_error("DetectionOutput layer supports only 3 or 5 inputs");

            ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(detOut)};
            function = std::make_shared<ngraph::Function>(results, params, "DetectionOutputDynamic");
        } else {
            auto detOut = ngraph::builder::makeDetectionOutput(paramOuts, attrs);
            ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(detOut)};
            function = std::make_shared<ngraph::Function>(results, params, "DetectionOutputDynamic");
        }
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

                input_shape.first[dimension] = {
                        interval_min,
                        interval_min == interval_max ? (interval_max + 1) : interval_max };
            }
        }
    }
    ngraph::op::DetectionOutputAttrs attrs;
    std::vector<ov::test::InputShape> inShapes;
};

TEST_P(DetectionOutputLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

const std::vector<int> numClasses = {11, -1};
const int backgroundLabelId = 0;
const std::vector<int> topK = {75};
const std::vector<std::vector<int>> keepTopK = { {50}, {100} };
const std::vector<std::string> codeType = {"caffe.PriorBoxParameter.CORNER", "caffe.PriorBoxParameter.CENTER_SIZE"};
const float nmsThreshold = 0.5f;
const float confidenceThreshold = 0.3f;
const std::vector<bool> clipAfterNms = {true, false};
const std::vector<bool> clipBeforeNms = {true, false};
const std::vector<bool> decreaseLabelId = {true, false};
const float objectnessScore = 0.4f;
const std::vector<size_t> numberBatch = {1, 2};

const auto commonAttributes1 = ::testing::Combine(
    ::testing::Values(numClasses[0]),
    ::testing::Values(backgroundLabelId),
    ::testing::ValuesIn(topK),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(codeType),
    ::testing::Values(nmsThreshold),
    ::testing::Values(confidenceThreshold),
    ::testing::ValuesIn(clipAfterNms),
    ::testing::ValuesIn(clipBeforeNms),
    ::testing::ValuesIn(decreaseLabelId)
);
const auto commonAttributes2 = ::testing::Combine(
    ::testing::Values(numClasses[0]),
    ::testing::Values(backgroundLabelId),
    ::testing::Values(topK[0]),
    ::testing::Values(keepTopK[0]),
    ::testing::Values(codeType[0]),
    ::testing::Values(nmsThreshold),
    ::testing::Values(confidenceThreshold),
    ::testing::Values(clipAfterNms[0]),
    ::testing::Values(clipBeforeNms[0]),
    ::testing::Values(decreaseLabelId[0])
);
const auto commonAttributes_v8 = ::testing::Combine(
    ::testing::Values(numClasses[1]),
    ::testing::Values(backgroundLabelId),
    ::testing::Values(topK[0]),
    ::testing::Values(keepTopK[0]),
    ::testing::ValuesIn(codeType),
    ::testing::Values(nmsThreshold),
    ::testing::Values(confidenceThreshold),
    ::testing::Values(clipAfterNms[0]),
    ::testing::Values(clipBeforeNms[0]),
    ::testing::Values(decreaseLabelId[0])
);
/* =============== 3 inputs cases =============== */

const std::vector<ParamsWhichSizeDependsDynamic> specificParams3InDynamic = {
    // dynamic input shapes
    ParamsWhichSizeDependsDynamic {
        true, true, true, 1, 1,
        {
            // input model dynamic shapes
            {ov::Dimension::dynamic(), ov::Dimension::dynamic()},
            // input tensor shapes
            {{1, 60}, {1, 120}}
        },
        {
            // input model dynamic shapes
            {ov::Dimension::dynamic(), ov::Dimension::dynamic()},
            // input tensor shapes
            {{1, 165}, {1, 330}}
        },
        {
            // input model dynamic shapes
            {ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
            // input tensor shapes
            {{1, 1, 60}, {1, 1, 120}}
        },
        {},
        {}
    },
    ParamsWhichSizeDependsDynamic {
        true, false, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 60}, {1, 1, 120}}},
        {},
        {}
    },
    ParamsWhichSizeDependsDynamic {
        false, true, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {},
        {}
    },
    ParamsWhichSizeDependsDynamic {
        false, false, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {},
        {}
    },
    ParamsWhichSizeDependsDynamic {
        true, true, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {},
        {}
    },
    ParamsWhichSizeDependsDynamic {
        true, false, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {},
        {}
    },
    ParamsWhichSizeDependsDynamic {
        false, true, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {},
        {}
    },
    ParamsWhichSizeDependsDynamic {
        false, false, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {},
        {}
    },
};

const auto params3InputsDynamic1 = ::testing::Combine(
        commonAttributes1,
        ::testing::Values(specificParams3InDynamic[0]),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(objectnessScore),
        ::testing::Values(false, true),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

const auto params3InputsDynamic2 = ::testing::Combine(
        commonAttributes2,
        ::testing::ValuesIn(specificParams3InDynamic),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(objectnessScore),
        ::testing::Values(false, true),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

const auto params3InputsDynamic_v8 = ::testing::Combine(
        commonAttributes_v8,
        ::testing::Values(specificParams3InDynamic[0]),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(objectnessScore),
        ::testing::Values(true),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_GPUDetectionOutputDynamic3In1, DetectionOutputLayerGPUTest,
                         params3InputsDynamic1,
                         DetectionOutputLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPUDetectionOutputDynamic3In2, DetectionOutputLayerGPUTest,
                         params3InputsDynamic2,
                         DetectionOutputLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPUDetectionOutputV8Dynamic3In, DetectionOutputLayerGPUTest,
                         params3InputsDynamic_v8,
                         DetectionOutputLayerGPUTest::getTestCaseName);
} // namespace
} // namespace GPULayerTestsDefinitions
