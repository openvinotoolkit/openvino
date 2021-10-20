// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/detection_output.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct DetectionOutputParams {
    template <class IT>
    DetectionOutputParams(const int num_classes,
                          const int background_label_id,
                          const int top_k,
                          const bool variance_encoded_in_target,
                          const std::vector<int> keep_top_k,
                          const std::string code_type,
                          const bool share_location,
                          const float nms_threshold,
                          const float confidence_threshold,
                          const bool clip_after_nms,
                          const bool clip_before_nms,
                          const bool decrease_label_id,
                          const bool normalized,
                          const size_t input_height,
                          const size_t input_width,
                          const float objectness_score,
                          const size_t num_prior_boxes,
                          const size_t num_images,
                          const bool is_priors_patch_size_1,
                          const ov::element::Type& iType,
                          const std::vector<IT>& locValues,
                          const std::vector<IT>& confValues,
                          const std::vector<IT>& priorBoxesValues,
                          const std::vector<IT>& oValues,
                          const std::string& test_name = "")
        : inType(iType),
          locData(CreateTensor(iType, locValues)),
          confData(CreateTensor(iType, confValues)),
          priorBoxesData(CreateTensor(iType, priorBoxesValues)),
          refData(CreateTensor(iType, oValues)),
          testcaseName(test_name) {
              attrs.num_classes = num_classes;
              attrs.background_label_id = background_label_id;
              attrs.top_k = top_k;
              attrs.variance_encoded_in_target = variance_encoded_in_target;
              attrs.keep_top_k = keep_top_k;
              attrs.code_type = code_type;
              attrs.share_location = share_location;
              attrs.nms_threshold = nms_threshold;
              attrs.confidence_threshold = confidence_threshold;
              attrs.clip_after_nms = clip_after_nms;
              attrs.clip_before_nms = clip_before_nms;
              attrs.decrease_label_id = decrease_label_id;
              attrs.normalized = normalized;
              attrs.input_height = input_height;
              attrs.input_width = input_width;

              size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
              size_t prior_box_size = attrs.normalized ? 4 : 5;

              locShape = ov::Shape{num_images, num_prior_boxes * num_loc_classes * prior_box_size};
              confShape = ov::Shape{num_images, num_prior_boxes * attrs.num_classes};
              priorBoxesShape =
              ov::Shape{is_priors_patch_size_1 ? 1UL : num_images, attrs.variance_encoded_in_target ? 1UL : 2UL, num_prior_boxes * prior_box_size};
          }

    ov::op::v0::DetectionOutput::Attributes attrs;
    ov::PartialShape locShape;
    ov::PartialShape confShape;
    ov::PartialShape priorBoxesShape;
    ov::element::Type inType;
    ov::runtime::Tensor locData;
    ov::runtime::Tensor confData;
    ov::runtime::Tensor priorBoxesData;
    ov::runtime::Tensor refData;
    std::string testcaseName;
};

class ReferenceDetectionOutputLayerTest : public testing::TestWithParam<DetectionOutputParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.locData, params.confData, params.priorBoxesData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<DetectionOutputParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "locShape=" << param.locShape << "_";
        result << "confShape=" << param.confShape << "_";
        result << "priorBoxesShape=" << param.priorBoxesShape << "_";
        result << "iType=" << param.inType;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const DetectionOutputParams& params) {
        const auto loc = std::make_shared<op::v0::Parameter>(params.inType, params.locShape);
        const auto conf = std::make_shared<op::v0::Parameter>(params.inType, params.confShape);
        const auto priorBoxes = std::make_shared<op::v0::Parameter>(params.inType, params.priorBoxesShape);
        const auto DetectionOutput = std::make_shared<op::v0::DetectionOutput>(loc, conf, priorBoxes, params.attrs);
        return std::make_shared<ov::Function>(NodeVector {DetectionOutput}, ParameterVector {loc, conf, priorBoxes});
    }
};

TEST_P(ReferenceDetectionOutputLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<DetectionOutputParams> generateDetectionOutputFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DetectionOutputParams> detectionOutputParams {
        DetectionOutputParams(3,
                              -1,
                              -1,
                              true,
                              {2},
                              "caffe.PriorBoxParameter.CORNER",
                              false,
                              0.5,
                              0.3,
                              false,
                              true,
                              false,
                              true,
                              0,
                              0,
                              0,
                              2,
                              2,
                              true,
                              IN_ET,
                              std::vector<T>{
                                    // batch 0, class 0
                                    0.1, 0.1, 0.2, 0.2, 0.0, 0.1, 0.2, 0.15,
                                    // batch 0, class 1
                                    0.3, 0.2, 0.5, 0.3, 0.2, 0.1, 0.42, 0.66,
                                    // batch 0, class 2
                                    0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.33, 0.44,
                                    // batch 1, class 0
                                    0.2, 0.1, 0.4, 0.2, 0.1, 0.05, 0.2, 0.25,
                                    // batch 1, class 1
                                    0.1, 0.2, 0.5, 0.3, 0.1, 0.1, 0.12, 0.34,
                                    // batch 1, class 2
                                    0.25, 0.11, 0.4, 0.32, 0.2, 0.12, 0.38, 0.24},
                              std::vector<T>{
                                    // batch 0
                                    0.1, 0.9, 0.4, 0.7, 0, 0.2,
                                    // batch 1
                                    0.7, 0.8, 0.42, 0.33, 0.81, 0.2},
                              std::vector<T>{
                                    // prior box 0
                                    0.0, 0.5, 0.1, 0.2,
                                    // prior box 1
                                    0.0, 0.3, 0.1, 0.35},
                              std::vector<T>{
                                    0, 0, 0.7,  0.2,  0.4,  0.52, 1,    0, 1, 0.9, 0,   0.6,  0.3, 0.35,
                                    1, 1, 0.81, 0.25, 0.41, 0.5,  0.67, 1, 1, 0.8, 0.1, 0.55, 0.3, 0.45},
                                    "3_inputs"),
        DetectionOutputParams(3,
                              -1,
                              -1,
                              true,
                              {2},
                              "caffe.PriorBoxParameter.CORNER",
                              true,
                              0.5,
                              0.3,
                              false,
                              true,
                              false,
                              true,
                              0,
                              0,
                              0,
                              2,
                              2,
                              false,
                              IN_ET,
                              std::vector<T>{
                                    // batch 0
                                    0.1, 0.1, 0.2, 0.2, 0.0, 0.1, 0.2, 0.15,
                                    // batch 1
                                    0.2, 0.1, 0.4, 0.2, 0.1, 0.05, 0.2, 0.25},
                              std::vector<T>{
                                    // batch 0
                                    0.1, 0.9, 0.4, 0.7, 0, 0.2,
                                    // batch 1
                                    0.7, 0.8, 0.42, 0.33, 0.81, 0.2},
                              std::vector<T>{
                                    // batch 0
                                    0.0, 0.5, 0.1, 0.2, 0.0, 0.3, 0.1, 0.35,
                                    // batch 1
                                    0.33, 0.2, 0.52, 0.37, 0.22, 0.1, 0.32, 0.36},
                              std::vector<T>{
                                    0, 0, 0.7,  0,    0.4,  0.3,  0.5,  0, 1, 0.9, 0.1,  0.6, 0.3,  0.4,
                                    1, 1, 0.81, 0.32, 0.15, 0.52, 0.61, 1, 1, 0.8, 0.53, 0.3, 0.92, 0.57},
                                    "3_inputs_share_location"),
        DetectionOutputParams(3,
                              -1,
                              -1,
                              true,
                              {2},
                              "caffe.PriorBoxParameter.CORNER",
                              true,
                              0.5,
                              0.3,
                              false,
                              true,
                              false,
                              true,
                              0,
                              0,
                              0,
                              2,
                              2,
                              false,
                              IN_ET,
                              std::vector<T>{
                                    // batch 0
                                    0.1, 0.1, 0.2, 0.2, 0.0, 0.1, 0.2, 0.15,
                                    // batch 1
                                    0.2, 0.1, 0.4, 0.2, 0.1, 0.05, 0.2, 0.25},
                              std::vector<T>{
                                    // batch 0
                                    0.1, 0.9, 0.4, 0.7, 0, 0.2,
                                    // batch 1
                                    0.7, 0.8, 0.42, 0.33, 0.81, 0.2},
                              std::vector<T>{
                                    // batch 0
                                    0.0, 0.5, 0.1, 0.2, 0.0, 0.3, 0.1, 0.35,
                                    // batch 1
                                    0.33, 0.2, 0.52, 0.37, 0.22, 0.1, 0.32, 0.36},
                              std::vector<T>{
                                    0, 0, 0.7,  0,    0.4,  0.3,  0.5,  0, 1, 0.9, 0.1,  0.6, 0.3,  0.4,
                                    1, 1, 0.81, 0.32, 0.15, 0.52, 0.61, 1, 1, 0.8, 0.53, 0.3, 0.92, 0.57},
                                    "3_inputs_normalized"),
    };
    return detectionOutputParams;
}

std::vector<DetectionOutputParams> generateDetectionOutputCombinedParams() {
    const std::vector<std::vector<DetectionOutputParams>> detectionOutputTypeParams {
        generateDetectionOutputFloatParams<element::Type_t::f64>(),
        generateDetectionOutputFloatParams<element::Type_t::f32>(),
        generateDetectionOutputFloatParams<element::Type_t::f16>(),
        generateDetectionOutputFloatParams<element::Type_t::bf16>(),
        };
    std::vector<DetectionOutputParams> combinedParams;

    for (const auto& params : detectionOutputTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput_With_Hardcoded_Refs, ReferenceDetectionOutputLayerTest,
    testing::ValuesIn(generateDetectionOutputCombinedParams()), ReferenceDetectionOutputLayerTest::getTestCaseName);

} // namespace