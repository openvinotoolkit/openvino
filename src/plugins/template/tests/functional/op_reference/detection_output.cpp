// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/detection_output.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
size_t get_k(const size_t num_images,
             const size_t num_prior_boxes,
             const size_t num_classes,
             const int top_k,
             const std::vector<int>& keep_top_k) {
    if (keep_top_k[0] > 0)
        return num_images * keep_top_k[0];
    else if (keep_top_k[0] == -1 && top_k > 0)
        return num_images * top_k * num_classes;
    else
        return num_images * num_prior_boxes * num_classes;
}

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
          testcaseName(test_name) {
        attrs.num_classes = num_classes;
        attrs_v8.background_label_id = attrs.background_label_id = background_label_id;
        attrs_v8.top_k = attrs.top_k = top_k;
        attrs_v8.variance_encoded_in_target = attrs.variance_encoded_in_target = variance_encoded_in_target;
        attrs_v8.keep_top_k = attrs.keep_top_k = keep_top_k;
        attrs_v8.code_type = attrs.code_type = code_type;
        attrs_v8.share_location = attrs.share_location = share_location;
        attrs_v8.nms_threshold = attrs.nms_threshold = nms_threshold;
        attrs_v8.confidence_threshold = attrs.confidence_threshold = confidence_threshold;
        attrs_v8.clip_after_nms = attrs.clip_after_nms = clip_after_nms;
        attrs_v8.clip_before_nms = attrs.clip_before_nms = clip_before_nms;
        attrs_v8.decrease_label_id = attrs.decrease_label_id = decrease_label_id;
        attrs_v8.normalized = attrs.normalized = normalized;
        attrs_v8.input_height = attrs.input_height = input_height;
        attrs_v8.input_width = attrs.input_width = input_width;
        attrs_v8.objectness_score = attrs.objectness_score = objectness_score;

        size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
        size_t prior_box_size = attrs.normalized ? 4 : 5;

        locShape = ov::Shape{num_images, num_prior_boxes * num_loc_classes * prior_box_size};
        confShape = ov::Shape{num_images, num_prior_boxes * attrs.num_classes};
        priorBoxesShape = ov::Shape{is_priors_patch_size_1 ? 1UL : num_images,
                                    attrs.variance_encoded_in_target ? 1UL : 2UL,
                                    num_prior_boxes * prior_box_size};

        const auto k = get_k(num_images, num_prior_boxes, num_classes, top_k, keep_top_k);
        const auto output_shape = Shape{1, 1, k, 7};
        refData = CreateTensor(output_shape, iType, oValues);
    }

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
                          const std::vector<IT>& auxLocValues,
                          const std::vector<IT>& auxConfValues,
                          const std::string& test_name = "")
        : inType(iType),
          locData(CreateTensor(iType, locValues)),
          confData(CreateTensor(iType, confValues)),
          priorBoxesData(CreateTensor(iType, priorBoxesValues)),
          auxLocData(CreateTensor(iType, auxLocValues)),
          auxConfData(CreateTensor(iType, auxConfValues)),
          testcaseName(test_name) {
        attrs.num_classes = num_classes;
        attrs_v8.background_label_id = attrs.background_label_id = background_label_id;
        attrs_v8.top_k = attrs.top_k = top_k;
        attrs_v8.variance_encoded_in_target = attrs.variance_encoded_in_target = variance_encoded_in_target;
        attrs_v8.keep_top_k = attrs.keep_top_k = keep_top_k;
        attrs_v8.code_type = attrs.code_type = code_type;
        attrs_v8.share_location = attrs.share_location = share_location;
        attrs_v8.nms_threshold = attrs.nms_threshold = nms_threshold;
        attrs_v8.confidence_threshold = attrs.confidence_threshold = confidence_threshold;
        attrs_v8.clip_after_nms = attrs.clip_after_nms = clip_after_nms;
        attrs_v8.clip_before_nms = attrs.clip_before_nms = clip_before_nms;
        attrs_v8.decrease_label_id = attrs.decrease_label_id = decrease_label_id;
        attrs_v8.normalized = attrs.normalized = normalized;
        attrs_v8.input_height = attrs.input_height = input_height;
        attrs_v8.input_width = attrs.input_width = input_width;
        attrs_v8.objectness_score = attrs.objectness_score = objectness_score;

        size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
        size_t prior_box_size = attrs.normalized ? 4 : 5;

        locShape = ov::Shape{num_images, num_prior_boxes * num_loc_classes * prior_box_size};
        confShape = ov::Shape{num_images, num_prior_boxes * attrs.num_classes};
        priorBoxesShape = ov::Shape{is_priors_patch_size_1 ? 1UL : num_images,
                                    attrs.variance_encoded_in_target ? 1UL : 2UL,
                                    num_prior_boxes * prior_box_size};
        auxLocShape = locShape;
        auxConfShape = confShape;

        const auto k = get_k(num_images, num_prior_boxes, num_classes, top_k, keep_top_k);
        const auto output_shape = Shape{1, 1, k, 7};
        refData = CreateTensor(output_shape, iType, oValues);
    }

    ov::op::v0::DetectionOutput::Attributes attrs;
    ov::op::v8::DetectionOutput::Attributes attrs_v8;
    ov::PartialShape locShape;
    ov::PartialShape confShape;
    ov::PartialShape priorBoxesShape;
    ov::PartialShape auxLocShape;
    ov::PartialShape auxConfShape;
    ov::element::Type inType;
    ov::Tensor locData;
    ov::Tensor confData;
    ov::Tensor priorBoxesData;
    ov::Tensor refData;
    ov::Tensor auxLocData;
    ov::Tensor auxConfData;
    std::string testcaseName;
};

class ReferenceDetectionOutputLayerTest : public testing::TestWithParam<DetectionOutputParams>,
                                          public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        if ((params.auxLocShape.size() != 0) && (params.auxConfShape.size() != 0))
            inputData = {params.locData, params.confData, params.priorBoxesData, params.auxConfData, params.auxLocData};
        else
            inputData = {params.locData, params.confData, params.priorBoxesData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<DetectionOutputParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "locShape=" << param.locShape << "_";
        result << "confShape=" << param.confShape << "_";
        result << "priorBoxesShape=" << param.priorBoxesShape << "_";
        if ((param.auxLocShape.size() != 0) && (param.auxConfShape.size() != 0)) {
            result << "auxLocShape=" << param.locShape << "_";
            result << "auxConfShape=" << param.confShape << "_";
        }
        result << "iType=" << param.inType;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const DetectionOutputParams& params) {
        const auto loc = std::make_shared<op::v0::Parameter>(params.inType, params.locShape);
        const auto conf = std::make_shared<op::v0::Parameter>(params.inType, params.confShape);
        const auto priorBoxes = std::make_shared<op::v0::Parameter>(params.inType, params.priorBoxesShape);
        if ((params.auxLocShape.size() != 0) && (params.auxConfShape.size() != 0)) {
            const auto auxConf = std::make_shared<op::v0::Parameter>(params.inType, params.auxConfShape);
            const auto auxLoc = std::make_shared<op::v0::Parameter>(params.inType, params.auxLocShape);
            const auto DetectionOutput =
                std::make_shared<op::v0::DetectionOutput>(loc, conf, priorBoxes, auxConf, auxLoc, params.attrs);
            return std::make_shared<ov::Model>(NodeVector{DetectionOutput},
                                               ParameterVector{loc, conf, priorBoxes, auxConf, auxLoc});
        } else {
            const auto DetectionOutput = std::make_shared<op::v0::DetectionOutput>(loc, conf, priorBoxes, params.attrs);
            return std::make_shared<ov::Model>(NodeVector{DetectionOutput}, ParameterVector{loc, conf, priorBoxes});
        }
    }
};

class ReferenceDetectionOutputV8LayerTest : public testing::TestWithParam<DetectionOutputParams>,
                                            public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        if ((params.auxLocShape.size() != 0) && (params.auxConfShape.size() != 0))
            inputData = {params.locData, params.confData, params.priorBoxesData, params.auxConfData, params.auxLocData};
        else
            inputData = {params.locData, params.confData, params.priorBoxesData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<DetectionOutputParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "locShape=" << param.locShape << "_";
        result << "confShape=" << param.confShape << "_";
        result << "priorBoxesShape=" << param.priorBoxesShape << "_";
        if ((param.auxLocShape.size() != 0) && (param.auxConfShape.size() != 0)) {
            result << "auxLocShape=" << param.locShape << "_";
            result << "auxConfShape=" << param.confShape << "_";
        }
        result << "iType=" << param.inType;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const DetectionOutputParams& params) {
        const auto loc = std::make_shared<op::v0::Parameter>(params.inType, params.locShape);
        const auto conf = std::make_shared<op::v0::Parameter>(params.inType, params.confShape);
        const auto priorBoxes = std::make_shared<op::v0::Parameter>(params.inType, params.priorBoxesShape);
        if ((params.auxLocShape.size() != 0) && (params.auxConfShape.size() != 0)) {
            const auto auxConf = std::make_shared<op::v0::Parameter>(params.inType, params.auxConfShape);
            const auto auxLoc = std::make_shared<op::v0::Parameter>(params.inType, params.auxLocShape);
            const auto DetectionOutput =
                std::make_shared<op::v8::DetectionOutput>(loc, conf, priorBoxes, auxConf, auxLoc, params.attrs_v8);
            return std::make_shared<ov::Model>(NodeVector{DetectionOutput},
                                               ParameterVector{loc, conf, priorBoxes, auxConf, auxLoc});
        } else {
            const auto DetectionOutput =
                std::make_shared<op::v8::DetectionOutput>(loc, conf, priorBoxes, params.attrs_v8);
            return std::make_shared<ov::Model>(NodeVector{DetectionOutput}, ParameterVector{loc, conf, priorBoxes});
        }
    }
};

TEST_P(ReferenceDetectionOutputLayerTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceDetectionOutputV8LayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<DetectionOutputParams> generateDetectionOutputFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DetectionOutputParams> detectionOutputParams{
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
                              std::vector<T>{// batch 0, class 0
                                             0.1,
                                             0.1,
                                             0.2,
                                             0.2,
                                             0.0,
                                             0.1,
                                             0.2,
                                             0.15,
                                             // batch 0, class 1
                                             0.3,
                                             0.2,
                                             0.5,
                                             0.3,
                                             0.2,
                                             0.1,
                                             0.42,
                                             0.66,
                                             // batch 0, class 2
                                             0.05,
                                             0.1,
                                             0.2,
                                             0.3,
                                             0.2,
                                             0.1,
                                             0.33,
                                             0.44,
                                             // batch 1, class 0
                                             0.2,
                                             0.1,
                                             0.4,
                                             0.2,
                                             0.1,
                                             0.05,
                                             0.2,
                                             0.25,
                                             // batch 1, class 1
                                             0.1,
                                             0.2,
                                             0.5,
                                             0.3,
                                             0.1,
                                             0.1,
                                             0.12,
                                             0.34,
                                             // batch 1, class 2
                                             0.25,
                                             0.11,
                                             0.4,
                                             0.32,
                                             0.2,
                                             0.12,
                                             0.38,
                                             0.24},
                              std::vector<T>{// batch 0
                                             0.1,
                                             0.9,
                                             0.4,
                                             0.7,
                                             0,
                                             0.2,
                                             // batch 1
                                             0.7,
                                             0.8,
                                             0.42,
                                             0.33,
                                             0.81,
                                             0.2},
                              std::vector<T>{// prior box 0
                                             0.0,
                                             0.5,
                                             0.1,
                                             0.2,
                                             // prior box 1
                                             0.0,
                                             0.3,
                                             0.1,
                                             0.35},
                              std::vector<T>{0, 0, 0.7,  0.2,  0.4,  0.52, 1,    0, 1, 0.9, 0,   0.6,  0.3, 0.35,
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
                              std::vector<T>{// batch 0
                                             0.1,
                                             0.1,
                                             0.2,
                                             0.2,
                                             0.0,
                                             0.1,
                                             0.2,
                                             0.15,
                                             // batch 1
                                             0.2,
                                             0.1,
                                             0.4,
                                             0.2,
                                             0.1,
                                             0.05,
                                             0.2,
                                             0.25},
                              std::vector<T>{// batch 0
                                             0.1,
                                             0.9,
                                             0.4,
                                             0.7,
                                             0,
                                             0.2,
                                             // batch 1
                                             0.7,
                                             0.8,
                                             0.42,
                                             0.33,
                                             0.81,
                                             0.2},
                              std::vector<T>{// batch 0
                                             0.0,
                                             0.5,
                                             0.1,
                                             0.2,
                                             0.0,
                                             0.3,
                                             0.1,
                                             0.35,
                                             // batch 1
                                             0.33,
                                             0.2,
                                             0.52,
                                             0.37,
                                             0.22,
                                             0.1,
                                             0.32,
                                             0.36},
                              std::vector<T>{0, 0, 0.7,  0,    0.4,  0.3,  0.5,  0, 1, 0.9, 0.1,  0.6, 0.3,  0.4,
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
                              std::vector<T>{// batch 0
                                             0.1,
                                             0.1,
                                             0.2,
                                             0.2,
                                             0.0,
                                             0.1,
                                             0.2,
                                             0.15,
                                             // batch 1
                                             0.2,
                                             0.1,
                                             0.4,
                                             0.2,
                                             0.1,
                                             0.05,
                                             0.2,
                                             0.25},
                              std::vector<T>{// batch 0
                                             0.1,
                                             0.9,
                                             0.4,
                                             0.7,
                                             0,
                                             0.2,
                                             // batch 1
                                             0.7,
                                             0.8,
                                             0.42,
                                             0.33,
                                             0.81,
                                             0.2},
                              std::vector<T>{// batch 0
                                             0.0,
                                             0.5,
                                             0.1,
                                             0.2,
                                             0.0,
                                             0.3,
                                             0.1,
                                             0.35,
                                             // batch 1
                                             0.33,
                                             0.2,
                                             0.52,
                                             0.37,
                                             0.22,
                                             0.1,
                                             0.32,
                                             0.36},
                              std::vector<T>{0, 0, 0.7,  0,    0.4,  0.3,  0.5,  0, 1, 0.9, 0.1,  0.6, 0.3,  0.4,
                                             1, 1, 0.81, 0.32, 0.15, 0.52, 0.61, 1, 1, 0.8, 0.53, 0.3, 0.92, 0.57},
                              "3_inputs_normalized"),
        DetectionOutputParams(
            2,
            -1,
            -1,
            false,
            {-1},
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
            3,
            false,
            IN_ET,
            std::vector<T>{// batch 0, class 0
                           0.1,
                           0.1,
                           0.2,
                           0.2,
                           0.0,
                           0.1,
                           0.2,
                           0.15,
                           // batch 0, class 1
                           0.3,
                           0.2,
                           0.5,
                           0.3,
                           0.2,
                           0.1,
                           0.42,
                           0.66,
                           // batch 1, class 0
                           0.05,
                           0.1,
                           0.2,
                           0.3,
                           0.2,
                           0.1,
                           0.33,
                           0.44,
                           // batch 1, class 1
                           0.2,
                           0.1,
                           0.4,
                           0.2,
                           0.1,
                           0.05,
                           0.2,
                           0.25,
                           // batch 2, class 0
                           0.1,
                           0.2,
                           0.5,
                           0.3,
                           0.1,
                           0.1,
                           0.12,
                           0.34,
                           // batch 2, class 1
                           0.25,
                           0.11,
                           0.4,
                           0.32,
                           0.2,
                           0.12,
                           0.38,
                           0.24},
            std::vector<T>{// batch 0
                           0.1,
                           0.9,
                           0.4,
                           0.7,
                           // batch 1
                           0.7,
                           0.8,
                           0.42,
                           0.33,
                           // batch 1
                           0.1,
                           0.2,
                           0.32,
                           0.43},
            std::vector<T>{// batch 0 priors
                           0.0,
                           0.5,
                           0.1,
                           0.2,
                           0.0,
                           0.3,
                           0.1,
                           0.35,
                           // batch 0 variances
                           0.12,
                           0.11,
                           0.32,
                           0.02,
                           0.02,
                           0.20,
                           0.09,
                           0.71,
                           // batch 1 priors
                           0.33,
                           0.2,
                           0.52,
                           0.37,
                           0.22,
                           0.1,
                           0.32,
                           0.36,
                           // batch 1 variances
                           0.01,
                           0.07,
                           0.12,
                           0.13,
                           0.41,
                           0.33,
                           0.2,
                           0.1,
                           // batch 2 priors
                           0.0,
                           0.3,
                           0.1,
                           0.35,
                           0.22,
                           0.1,
                           0.32,
                           0.36,
                           // batch 2 variances
                           0.32,
                           0.02,
                           0.13,
                           0.41,
                           0.33,
                           0.2,
                           0.02,
                           0.20},
            std::vector<T>{0, 0, 0.4,  0.006, 0.34,   0.145,  0.563,  0,  1, 0.9,  0,      0.511, 0.164,  0.203,
                           0, 1, 0.7,  0.004, 0.32,   0.1378, 0.8186, 1,  0, 0.7,  0.3305, 0.207, 0.544,  0.409,
                           1, 0, 0.42, 0.302, 0.133,  0.4,    0.38,   1,  1, 0.8,  0.332,  0.207, 0.5596, 0.4272,
                           1, 1, 0.33, 0.261, 0.1165, 0.36,   0.385,  2,  0, 0.32, 0.3025, 0.122, 0.328,  0.424,
                           2, 1, 0.43, 0.286, 0.124,  0.3276, 0.408,  -1, 0, 0,    0,      0,     0,      0,
                           0, 0, 0,    0,     0,      0,      0,      0,  0, 0,    0,      0,     0,      0},
            "3_inputs_keep_all_bboxes"),
        DetectionOutputParams(3,
                              -1,
                              -1,
                              true,
                              {2},
                              "caffe.PriorBoxParameter.CENTER_SIZE",
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
                              false,
                              IN_ET,
                              std::vector<T>{// batch 0, class 0
                                             0.1,
                                             0.1,
                                             0.2,
                                             0.2,
                                             0.0,
                                             0.1,
                                             0.2,
                                             0.15,
                                             // batch 0, class 1
                                             0.3,
                                             0.2,
                                             0.5,
                                             0.3,
                                             0.2,
                                             0.1,
                                             0.42,
                                             0.66,
                                             // batch 0, class 2
                                             0.05,
                                             0.1,
                                             0.2,
                                             0.3,
                                             0.2,
                                             0.1,
                                             0.33,
                                             0.44,
                                             // batch 1, class 0
                                             0.2,
                                             0.1,
                                             0.4,
                                             0.2,
                                             0.1,
                                             0.05,
                                             0.2,
                                             0.25,
                                             // batch 1, class 1
                                             0.1,
                                             0.2,
                                             0.5,
                                             0.3,
                                             0.1,
                                             0.1,
                                             0.12,
                                             0.34,
                                             // batch 1, class 2
                                             0.25,
                                             0.11,
                                             0.4,
                                             0.32,
                                             0.2,
                                             0.12,
                                             0.38,
                                             0.24},
                              std::vector<T>{// batch 0
                                             0.1,
                                             0.9,
                                             0.4,
                                             0.7,
                                             0,
                                             0.2,
                                             // batch 1
                                             0.7,
                                             0.8,
                                             0.42,
                                             0.33,
                                             0.81,
                                             0.2},
                              std::vector<T>{// batch 0
                                             0.0,
                                             0.5,
                                             0.1,
                                             0.2,
                                             0.0,
                                             0.3,
                                             0.1,
                                             0.35,
                                             // batch 1
                                             0.33,
                                             0.2,
                                             0.52,
                                             0.37,
                                             0.22,
                                             0.1,
                                             0.32,
                                             0.36},
                              std::vector<T>{0, 0, 0.7,  0,          0.28163019,  0.14609808, 0.37836978,
                                             0, 1, 0.9,  0,          0.49427515,  0.11107014, 0.14572485,
                                             1, 1, 0.81, 0.22040875, 0.079573378, 0.36959124, 0.4376266,
                                             1, 1, 0.8,  0.32796675, 0.18435785,  0.56003326, 0.40264216},
                              "3_inputs_center_size"),
        DetectionOutputParams(2,
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
                              0.6,
                              2,
                              2,
                              false,
                              IN_ET,
                              std::vector<T>{// batch 0, class 0
                                             0.1,
                                             0.1,
                                             0.2,
                                             0.2,
                                             0.0,
                                             0.1,
                                             0.2,
                                             0.15,
                                             // batch 0, class 1
                                             0.3,
                                             0.2,
                                             0.5,
                                             0.3,
                                             0.2,
                                             0.1,
                                             0.42,
                                             0.66,
                                             // batch 1, class 0
                                             0.2,
                                             0.1,
                                             0.4,
                                             0.2,
                                             0.1,
                                             0.05,
                                             0.2,
                                             0.25,
                                             // batch 1, class 1
                                             0.1,
                                             0.2,
                                             0.5,
                                             0.3,
                                             0.1,
                                             0.1,
                                             0.12,
                                             0.34},
                              std::vector<T>{// batch 0
                                             0.1,
                                             0.9,
                                             0.4,
                                             0.7,
                                             // batch 1
                                             0.42,
                                             0.33,
                                             0.81,
                                             0.2},
                              std::vector<T>{// batch 0
                                             0.0,
                                             0.5,
                                             0.1,
                                             0.2,
                                             0.0,
                                             0.3,
                                             0.1,
                                             0.35,
                                             // batch 1
                                             0.33,
                                             0.2,
                                             0.52,
                                             0.37,
                                             0.22,
                                             0.1,
                                             0.32,
                                             0.36},
                              std::vector<T>{0, 0, 0.4,  0.55, 0.61, 1, 0.97, 0, 1, 0.7,  0.4,  0.52, 0.9, 1,
                                             1, 0, 0.42, 0.83, 0.5,  1, 0.87, 1, 1, 0.33, 0.63, 0.35, 1,   1},
                              std::vector<T>{// batch 0, class 0
                                             0.1,
                                             0.2,
                                             0.5,
                                             0.3,
                                             0.1,
                                             0.1,
                                             0.12,
                                             0.34,
                                             // batch 0, class 1
                                             0.25,
                                             0.11,
                                             0.4,
                                             0.32,
                                             0.2,
                                             0.12,
                                             0.38,
                                             0.24,
                                             // batch 1, class 0
                                             0.3,
                                             0.2,
                                             0.5,
                                             0.3,
                                             0.2,
                                             0.1,
                                             0.42,
                                             0.66,
                                             // batch 1, class 1
                                             0.05,
                                             0.1,
                                             0.2,
                                             0.3,
                                             0.2,
                                             0.1,
                                             0.33,
                                             0.44},
                              std::vector<T>{// batch 0
                                             0.1,
                                             0.3,
                                             0.5,
                                             0.8,
                                             // batch 1
                                             0.5,
                                             0.8,
                                             0.01,
                                             0.1},
                              "5_inputs"),
    };
    return detectionOutputParams;
}

std::vector<DetectionOutputParams> generateDetectionOutputCombinedParams() {
    const std::vector<std::vector<DetectionOutputParams>> detectionOutputTypeParams{
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

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput_With_Hardcoded_Refs,
                         ReferenceDetectionOutputLayerTest,
                         testing::ValuesIn(generateDetectionOutputCombinedParams()),
                         ReferenceDetectionOutputLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput_With_Hardcoded_Refs,
                         ReferenceDetectionOutputV8LayerTest,
                         testing::ValuesIn(generateDetectionOutputCombinedParams()),
                         ReferenceDetectionOutputV8LayerTest::getTestCaseName);

}  // namespace
