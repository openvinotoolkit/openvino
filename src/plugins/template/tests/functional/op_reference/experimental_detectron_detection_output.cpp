// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_detection_output.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

using Attrs = op::v6::ExperimentalDetectronDetectionOutput::Attributes;

namespace {
struct ExperimentalDOParams {
    template <class IT>
    ExperimentalDOParams(const Attrs& attrs,
                         const size_t num_rois,
                         const element::Type& iType,
                         const std::vector<IT>& roisValues,
                         const std::vector<IT>& deltasValues,
                         const std::vector<IT>& scoresValues,
                         const std::vector<IT>& imageSizeInfoValues,
                         const std::vector<IT>& refBoxesValues,
                         const std::vector<int32_t>& refClassesValues,
                         const std::vector<IT>& refScoresValues,
                         const std::string& testcaseName = "")
        : attrs(attrs),
          inType(iType),
          outType(iType),
          roisData(CreateTensor(iType, roisValues)),
          deltasData(CreateTensor(iType, deltasValues)),
          scoresData(CreateTensor(iType, scoresValues)),
          imageSizeInfoData(CreateTensor(iType, imageSizeInfoValues)),
          testcaseName(testcaseName) {
        roisShape = Shape{num_rois, 4};
        deltasShape = Shape{num_rois, static_cast<size_t>(attrs.num_classes * 4)};
        scoresShape = Shape{num_rois, static_cast<size_t>(attrs.num_classes)};
        imageSizeInfoShape = Shape{1, 3};

        const auto max_d = attrs.max_detections_per_image;
        refBoxesData = CreateTensor(Shape{max_d, 4}, iType, refBoxesValues);
        refClassesData = CreateTensor(Shape{max_d}, ov::element::i32, refClassesValues);
        refScoresData = CreateTensor(Shape{max_d}, iType, refScoresValues);
    }

    Attrs attrs;
    PartialShape roisShape;
    PartialShape deltasShape;
    PartialShape scoresShape;
    PartialShape imageSizeInfoShape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor roisData;
    ov::Tensor deltasData;
    ov::Tensor scoresData;
    ov::Tensor imageSizeInfoData;
    ov::Tensor refBoxesData;
    ov::Tensor refClassesData;
    ov::Tensor refScoresData;
    std::string testcaseName;
};

class ReferenceExperimentalDOLayerTest : public testing::TestWithParam<ExperimentalDOParams>,
                                         public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.roisData, params.deltasData, params.scoresData, params.imageSizeInfoData};
        refOutData = {params.refBoxesData, params.refClassesData, params.refScoresData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDOParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "roisShape=" << param.roisShape << "_";
        result << "deltasShape=" << param.deltasShape << "_";
        result << "scoresShape=" << param.scoresShape << "_";
        result << "imageSizeInfoShape=" << param.imageSizeInfoShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ExperimentalDOParams& params) {
        const auto rois = std::make_shared<op::v0::Parameter>(params.inType, params.roisShape);
        const auto deltas = std::make_shared<op::v0::Parameter>(params.inType, params.deltasShape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.inType, params.scoresShape);
        const auto im_info = std::make_shared<op::v0::Parameter>(params.inType, params.imageSizeInfoShape);
        const auto ExperimentalDO =
            std::make_shared<op::v6::ExperimentalDetectronDetectionOutput>(rois, deltas, scores, im_info, params.attrs);
        return std::make_shared<ov::Model>(ExperimentalDO->outputs(), ParameterVector{rois, deltas, scores, im_info});
    }
};

TEST_P(ReferenceExperimentalDOLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ExperimentalDOParams> generateExperimentalDOFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ExperimentalDOParams> experimentalDOParams{
        ExperimentalDOParams(
            Attrs{
                0.01000000074505806f,       // score_threshold
                0.2f,                       // nms_threshold
                2.0f,                       // max_delta_log_wh
                2,                          // num_classes
                500,                        // post_nms_count
                5,                          // max_detections_per_image
                true,                       // class_agnostic_box_regression
                {10.0f, 10.0f, 5.0f, 5.0f}  // deltas_weights
            },
            16,
            IN_ET,
            std::vector<T>{1.0f, 1.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f,  4.0f,  1.0f, 8.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
            std::vector<T>{
                5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
            std::vector<T>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
            std::vector<T>{1.0f, 1.0f, 1.0f},
            std::vector<T>{0.8929862f,
                           0.892986297607421875,
                           12.10701370239257812,
                           12.10701370239257812,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0f,
                           0.0},
            std::vector<int32_t>{1, 0, 0, 0, 0},
            std::vector<T>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f}),
    };
    return experimentalDOParams;
}

std::vector<ExperimentalDOParams> generateExperimentalDOCombinedParams() {
    const std::vector<std::vector<ExperimentalDOParams>> ExperimentalDOTypeParams{
        generateExperimentalDOFloatParams<element::Type_t::f32>(),
        generateExperimentalDOFloatParams<element::Type_t::f16>(),
        generateExperimentalDOFloatParams<element::Type_t::bf16>(),
    };
    std::vector<ExperimentalDOParams> combinedParams;

    for (const auto& params : ExperimentalDOTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronDetectionOutput_With_Hardcoded_Refs,
                         ReferenceExperimentalDOLayerTest,
                         testing::ValuesIn(generateExperimentalDOCombinedParams()),
                         ReferenceExperimentalDOLayerTest::getTestCaseName);
}  // namespace
