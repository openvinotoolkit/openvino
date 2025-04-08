// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roi_align_rotated.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

constexpr double PI = 3.141592653589793238462643383279;

struct ROIAlignRotatedParams {
    PartialShape inputShape;
    int32_t pooledH;
    int32_t pooledW;
    float spatialScale;
    int32_t samplingRatio;
    bool clockwise;
    std::string testcaseName;
    ov::Tensor input;
    reference_tests::Tensor rois;
    reference_tests::Tensor roiBatchIdxs;
    reference_tests::Tensor expectedOutput;
};

template <typename T>
ROIAlignRotatedParams PrepareTestCaseParams(const PartialShape& inputShape,
                                            size_t pooledH,
                                            size_t pooledW,
                                            float spatialScale,
                                            int32_t samplingRatio,
                                            bool clockwise,
                                            const std::vector<T>& inputValues,
                                            const std::vector<T>& roisVals,
                                            const std::vector<int32_t>& roiBatchIdx,
                                            const std::vector<T>& expectedValues,
                                            const std::string& testcaseName) {
    ROIAlignRotatedParams ret;

    constexpr size_t rois_second_dim_size = 5;  //< By definition of the ROIAlignRotated op

    const auto numOfRois = roisVals.size() / rois_second_dim_size;
    const auto channels = static_cast<size_t>(inputShape[1].get_length());
    const auto elementType = element::from<T>();

    ret.inputShape = inputShape;
    ret.pooledH = pooledH;
    ret.pooledW = pooledW;
    ret.spatialScale = spatialScale;
    ret.samplingRatio = samplingRatio;
    ret.clockwise = clockwise;
    ret.testcaseName = testcaseName;
    ret.input = CreateTensor(elementType, inputValues);
    ret.rois = reference_tests::Tensor(elementType, {numOfRois, 5}, roisVals);
    ret.roiBatchIdxs = reference_tests::Tensor(element::Type_t::i32, {numOfRois}, roiBatchIdx);
    ret.expectedOutput = reference_tests::Tensor(elementType, {numOfRois, channels, pooledH, pooledW}, expectedValues);

    return ret;
}

class ReferenceROIAlignRotatedTest : public testing::TestWithParam<ROIAlignRotatedParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.input};
        refOutData = {params.expectedOutput.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ROIAlignRotatedParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "type=" << param.input.get_element_type();
        result << "_inputShape=" << param.inputShape;
        result << "_pooledH=" << param.pooledH;
        result << "_pooledW=" << param.pooledW;
        result << "_spatialScale=" << param.spatialScale;
        result << "_samplingRatio=" << param.samplingRatio;
        result << "_clockwise=" << param.clockwise;
        result << "_roisShape=" << param.rois.shape;
        result << "_roisBatchIdxShape=" << param.roiBatchIdxs.shape;
        result << "_outputShape=" << param.expectedOutput.shape;
        if (!param.testcaseName.empty()) {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ROIAlignRotatedParams& params) {
        const auto featureMap = std::make_shared<op::v0::Parameter>(params.input.get_element_type(), params.inputShape);
        const auto coords =
            std::make_shared<op::v0::Constant>(params.rois.type, params.rois.shape, params.rois.data.data());
        const auto roisIdx = std::make_shared<op::v0::Constant>(params.roiBatchIdxs.type,
                                                                params.roiBatchIdxs.shape,
                                                                params.roiBatchIdxs.data.data());
        const auto roi_align_rot = std::make_shared<op::v15::ROIAlignRotated>(featureMap,
                                                                              coords,
                                                                              roisIdx,
                                                                              params.pooledH,
                                                                              params.pooledW,
                                                                              params.samplingRatio,
                                                                              params.spatialScale,
                                                                              params.clockwise);
        return std::make_shared<Model>(NodeVector{roi_align_rot}, ParameterVector{featureMap});
    }
};

TEST_P(ReferenceROIAlignRotatedTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ROIAlignRotatedParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ROIAlignRotatedParams> params;

#define TEST_DATA(input_shape,                                    \
                  pooled_height,                                  \
                  pooled_width,                                   \
                  spatial_scale,                                  \
                  sampling_ratio,                                 \
                  clockwise,                                      \
                  input_data,                                     \
                  rois_data,                                      \
                  batch_indices_data,                             \
                  expected_output,                                \
                  description)                                    \
    params.push_back(PrepareTestCaseParams<T>(input_shape,        \
                                              pooled_height,      \
                                              pooled_width,       \
                                              spatial_scale,      \
                                              sampling_ratio,     \
                                              clockwise,          \
                                              input_data,         \
                                              rois_data,          \
                                              batch_indices_data, \
                                              expected_output,    \
                                              description));

#include "unit_test_utils/tests_data/roi_align_rotated_data.h"
#undef TEST_DATA

    return params;
}

std::vector<ROIAlignRotatedParams> generateCombinedParams() {
    const std::vector<std::vector<ROIAlignRotatedParams>> generatedParams{
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<ROIAlignRotatedParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ROIAlignRotated_With_Hardcoded_Refs,
                         ReferenceROIAlignRotatedTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceROIAlignRotatedTest::getTestCaseName);
}  // namespace
