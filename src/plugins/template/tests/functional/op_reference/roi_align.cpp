// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roi_align.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ROIAlignParams {
    template <class IT>
    ROIAlignParams(const PartialShape& pShape,
                   const element::Type& iType,
                   const std::vector<IT>& iValues,
                   const reference_tests::Tensor& expectedFeatureMap,
                   const reference_tests::Tensor& coords,
                   const reference_tests::Tensor& roiIdx,
                   const int32_t pooledH,
                   const int32_t pooledW,
                   const float spatialScale,
                   const int32_t poolingRatio,
                   const std::string& poolingMode,
                   const std::string& testcaseName)
        : pShape(pShape),
          iType(iType),
          featureMap(CreateTensor(iType, iValues)),
          expectedFeatureMap(expectedFeatureMap),
          coords(coords),
          roiIdx(roiIdx),
          pooledH(pooledH),
          pooledW(pooledW),
          spatialScale(spatialScale),
          poolingRatio(poolingRatio),
          poolingMode(poolingMode),
          testcaseName(testcaseName) {}

    PartialShape pShape;
    element::Type iType;
    ov::Tensor featureMap;
    reference_tests::Tensor expectedFeatureMap;
    reference_tests::Tensor coords;
    reference_tests::Tensor roiIdx;
    int32_t pooledH;
    int32_t pooledW;
    float spatialScale;
    int32_t poolingRatio;
    std::string poolingMode;
    std::string testcaseName;
};

struct ROIAlignV9Params {
    template <class IT>
    ROIAlignV9Params(const PartialShape& pShape,
                     const element::Type& iType,
                     const std::vector<IT>& iValues,
                     const reference_tests::Tensor& expectedFeatureMap,
                     const reference_tests::Tensor& coords,
                     const reference_tests::Tensor& roiIdx,
                     const int32_t pooledH,
                     const int32_t pooledW,
                     const float spatialScale,
                     const int32_t poolingRatio,
                     const std::string& poolingMode,
                     const std::string& alignedMode,
                     const std::string& testcaseName)
        : pShape(pShape),
          iType(iType),
          featureMap(CreateTensor(iType, iValues)),
          expectedFeatureMap(expectedFeatureMap),
          coords(coords),
          roiIdx(roiIdx),
          pooledH(pooledH),
          pooledW(pooledW),
          spatialScale(spatialScale),
          poolingRatio(poolingRatio),
          poolingMode(poolingMode),
          alignedMode(alignedMode),
          testcaseName(testcaseName) {}

    PartialShape pShape;
    element::Type iType;
    ov::Tensor featureMap;
    reference_tests::Tensor expectedFeatureMap;
    reference_tests::Tensor coords;
    reference_tests::Tensor roiIdx;
    int32_t pooledH;
    int32_t pooledW;
    float spatialScale;
    int32_t poolingRatio;
    std::string poolingMode;
    std::string alignedMode;
    std::string testcaseName;
};

class ReferenceROIAlignTest : public testing::TestWithParam<ROIAlignParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.featureMap};
        refOutData = {params.expectedFeatureMap.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ROIAlignParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.iType;
        result << "_pShape=" << param.pShape;
        result << "_efType=" << param.expectedFeatureMap.type;
        result << "_efShape=" << param.expectedFeatureMap.shape;
        result << "_cType=" << param.coords.type;
        result << "_cShape=" << param.coords.shape;
        result << "_rType=" << param.roiIdx.type;
        result << "_rShape=" << param.roiIdx.shape;
        result << "_pooledH=" << param.pooledH;
        result << "_pooledW=" << param.pooledW;
        result << "_spatialScale=" << param.spatialScale;
        result << "_poolingRatio=" << param.poolingRatio;
        result << "_poolingMode=" << param.poolingMode;
        if (param.testcaseName != "") {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ROIAlignParams& params) {
        const auto featureMap = std::make_shared<op::v0::Parameter>(params.iType, params.pShape);
        const auto coords =
            std::make_shared<op::v0::Constant>(params.coords.type, params.coords.shape, params.coords.data.data());
        const auto roisIdx =
            std::make_shared<op::v0::Constant>(params.roiIdx.type, params.roiIdx.shape, params.roiIdx.data.data());
        const auto roi_align = std::make_shared<op::v3::ROIAlign>(featureMap,
                                                                  coords,
                                                                  roisIdx,
                                                                  params.pooledH,
                                                                  params.pooledW,
                                                                  params.poolingRatio,
                                                                  params.spatialScale,
                                                                  params.poolingMode);
        auto f = std::make_shared<Model>(NodeVector{roi_align}, ParameterVector{featureMap});
        return f;
    }
};

class ReferenceROIAlignV9Test : public testing::TestWithParam<ROIAlignV9Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.featureMap};
        refOutData = {params.expectedFeatureMap.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ROIAlignV9Params>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.iType;
        result << "_pShape=" << param.pShape;
        result << "_efType=" << param.expectedFeatureMap.type;
        result << "_efShape=" << param.expectedFeatureMap.shape;
        result << "_cType=" << param.coords.type;
        result << "_cShape=" << param.coords.shape;
        result << "_rType=" << param.roiIdx.type;
        result << "_rShape=" << param.roiIdx.shape;
        result << "_pooledH=" << param.pooledH;
        result << "_pooledW=" << param.pooledW;
        result << "_spatialScale=" << param.spatialScale;
        result << "_poolingRatio=" << param.poolingRatio;
        result << "_poolingMode=" << param.poolingMode;
        result << "_alignedMode=" << param.alignedMode;
        if (param.testcaseName != "") {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ROIAlignV9Params& params) {
        const auto featureMap = std::make_shared<op::v0::Parameter>(params.iType, params.pShape);
        const auto coords =
            std::make_shared<op::v0::Constant>(params.coords.type, params.coords.shape, params.coords.data.data());
        const auto roisIdx =
            std::make_shared<op::v0::Constant>(params.roiIdx.type, params.roiIdx.shape, params.roiIdx.data.data());
        const auto pooling_mode = EnumNames<op::v9::ROIAlign::PoolingMode>::as_enum(params.poolingMode);
        const auto aligned_mode = EnumNames<op::v9::ROIAlign::AlignedMode>::as_enum(params.alignedMode);
        const auto roi_align = std::make_shared<op::v9::ROIAlign>(featureMap,
                                                                  coords,
                                                                  roisIdx,
                                                                  params.pooledH,
                                                                  params.pooledW,
                                                                  params.poolingRatio,
                                                                  params.spatialScale,
                                                                  pooling_mode,
                                                                  aligned_mode);
        auto f = std::make_shared<Model>(NodeVector{roi_align}, ParameterVector{featureMap});
        return f;
    }
};

TEST_P(ReferenceROIAlignTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceROIAlignV9Test, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_IND>
std::vector<ROIAlignParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<ROIAlignParams> params{
        ROIAlignParams(
            PartialShape{2, 1, 8, 8},
            ET,
            std::vector<T>{0, 1, 8, 5, 5,  2, 0,  7, 7, 10, 4,  5, 9,  0, 0, 5,  7,  0, 4, 0, 4, 7,  6,  10, 9,  5,
                           1, 7, 4, 7, 10, 8, 2,  0, 8, 3,  6,  8, 10, 4, 2, 10, 7,  8, 7, 0, 6, 9,  2,  4,  8,  5,
                           2, 3, 3, 1, 5,  9, 10, 0, 9, 5,  5,  3, 10, 5, 2, 0,  10, 0, 5, 4, 3, 10, 5,  5,  10, 0,
                           8, 8, 9, 1, 0,  7, 9,  6, 8, 7,  10, 9, 2,  3, 3, 5,  6,  9, 4, 9, 2, 4,  5,  5,  3,  1,
                           1, 6, 8, 0, 5,  5, 10, 8, 6, 9,  6,  9, 1,  2, 7, 1,  1,  3, 0, 4, 0, 7,  10, 2},
            reference_tests::Tensor(ET, {2, 1, 2, 2}, std::vector<T>{3, 3.75, 4.75, 5, 3, 5.5, 2.75, 3.75}),
            reference_tests::Tensor(ET, {2, 4}, std::vector<T>{2, 2, 4, 4, 2, 2, 4, 4}),
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{0, 1}),
            2,
            2,
            1,
            2,
            "avg",
            "roi_align_avg"),

        ROIAlignParams(
            PartialShape{2, 1, 8, 8},
            ET,
            std::vector<T>{0, 1, 8, 5, 5,  2, 0,  7, 7, 10, 4,  5, 9,  0, 0, 5,  7,  0, 4, 0, 4, 7,  6,  10, 9,  5,
                           1, 7, 4, 7, 10, 8, 2,  0, 8, 3,  6,  8, 10, 4, 2, 10, 7,  8, 7, 0, 6, 9,  2,  4,  8,  5,
                           2, 3, 3, 1, 5,  9, 10, 0, 9, 5,  5,  3, 10, 5, 2, 0,  10, 0, 5, 4, 3, 10, 5,  5,  10, 0,
                           8, 8, 9, 1, 0,  7, 9,  6, 8, 7,  10, 9, 2,  3, 3, 5,  6,  9, 4, 9, 2, 4,  5,  5,  3,  1,
                           1, 6, 8, 0, 5,  5, 10, 8, 6, 9,  6,  9, 1,  2, 7, 1,  1,  3, 0, 4, 0, 7,  10, 2},
            reference_tests::Tensor(ET,
                                    {2, 1, 2, 2},
                                    std::vector<T>{4.375, 4.9375, 5.6875, 5.625, 4.625, 7.125, 3.3125, 4.3125}),
            reference_tests::Tensor(ET, {2, 4}, std::vector<T>{2, 2, 4, 4, 2, 2, 4, 4}),
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{0, 1}),
            2,
            2,
            1,
            2,
            "max",
            "roi_align_max"),
    };
    return params;
}

std::vector<ROIAlignParams> generateCombinedParams() {
    const std::vector<std::vector<ROIAlignParams>> generatedParams{
        generateParams<element::Type_t::bf16, element::Type_t::i8>(),
        generateParams<element::Type_t::f16, element::Type_t::i8>(),
        generateParams<element::Type_t::f32, element::Type_t::i8>(),
        generateParams<element::Type_t::bf16, element::Type_t::i16>(),
        generateParams<element::Type_t::f16, element::Type_t::i16>(),
        generateParams<element::Type_t::f32, element::Type_t::i16>(),
        generateParams<element::Type_t::bf16, element::Type_t::i32>(),
        generateParams<element::Type_t::f16, element::Type_t::i32>(),
        generateParams<element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::bf16, element::Type_t::i64>(),
        generateParams<element::Type_t::f16, element::Type_t::i64>(),
        generateParams<element::Type_t::f32, element::Type_t::i64>(),
        generateParams<element::Type_t::bf16, element::Type_t::u8>(),
        generateParams<element::Type_t::f16, element::Type_t::u8>(),
        generateParams<element::Type_t::f32, element::Type_t::u8>(),
        generateParams<element::Type_t::bf16, element::Type_t::u16>(),
        generateParams<element::Type_t::f16, element::Type_t::u16>(),
        generateParams<element::Type_t::f32, element::Type_t::u16>(),
        generateParams<element::Type_t::bf16, element::Type_t::u32>(),
        generateParams<element::Type_t::f16, element::Type_t::u32>(),
        generateParams<element::Type_t::f32, element::Type_t::u32>(),
        generateParams<element::Type_t::bf16, element::Type_t::u64>(),
        generateParams<element::Type_t::f16, element::Type_t::u64>(),
        generateParams<element::Type_t::f32, element::Type_t::u64>(),
    };
    std::vector<ROIAlignParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET, element::Type_t ET_IND>
std::vector<ROIAlignV9Params> generateParamsV9() {
    using T = typename element_type_traits<ET>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<ROIAlignV9Params> params{
        ROIAlignV9Params(
            PartialShape{2, 1, 8, 8},
            ET,
            std::vector<T>{0, 1, 8, 5, 5,  2, 0,  7, 7, 10, 4,  5, 9,  0, 0, 5,  7,  0, 4, 0, 4, 7,  6,  10, 9,  5,
                           1, 7, 4, 7, 10, 8, 2,  0, 8, 3,  6,  8, 10, 4, 2, 10, 7,  8, 7, 0, 6, 9,  2,  4,  8,  5,
                           2, 3, 3, 1, 5,  9, 10, 0, 9, 5,  5,  3, 10, 5, 2, 0,  10, 0, 5, 4, 3, 10, 5,  5,  10, 0,
                           8, 8, 9, 1, 0,  7, 9,  6, 8, 7,  10, 9, 2,  3, 3, 5,  6,  9, 4, 9, 2, 4,  5,  5,  3,  1,
                           1, 6, 8, 0, 5,  5, 10, 8, 6, 9,  6,  9, 1,  2, 7, 1,  1,  3, 0, 4, 0, 7,  10, 2},
            reference_tests::Tensor(ET, {2, 1, 2, 2}, std::vector<T>{3, 3.75, 4.75, 5, 3, 5.5, 2.75, 3.75}),
            reference_tests::Tensor(ET, {2, 4}, std::vector<T>{2, 2, 4, 4, 2, 2, 4, 4}),
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{0, 1}),
            2,
            2,
            1,
            2,
            "avg",
            "asymmetric",
            "roi_align_v9_avg_asymmetric"),

        ROIAlignV9Params(
            PartialShape{2, 1, 8, 8},
            ET,
            std::vector<T>{0, 1, 8, 5, 5,  2, 0,  7, 7, 10, 4,  5, 9,  0, 0, 5,  7,  0, 4, 0, 4, 7,  6,  10, 9,  5,
                           1, 7, 4, 7, 10, 8, 2,  0, 8, 3,  6,  8, 10, 4, 2, 10, 7,  8, 7, 0, 6, 9,  2,  4,  8,  5,
                           2, 3, 3, 1, 5,  9, 10, 0, 9, 5,  5,  3, 10, 5, 2, 0,  10, 0, 5, 4, 3, 10, 5,  5,  10, 0,
                           8, 8, 9, 1, 0,  7, 9,  6, 8, 7,  10, 9, 2,  3, 3, 5,  6,  9, 4, 9, 2, 4,  5,  5,  3,  1,
                           1, 6, 8, 0, 5,  5, 10, 8, 6, 9,  6,  9, 1,  2, 7, 1,  1,  3, 0, 4, 0, 7,  10, 2},
            reference_tests::Tensor(ET, {2, 1, 2, 2}, std::vector<T>{3.14, 2.16, 2.86, 5.03, 1.83, 5.84, 2.77, 3.44}),
            reference_tests::Tensor(ET, {2, 4}, std::vector<T>{2, 2, 4, 4, 2, 2, 4, 4}),
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{0, 1}),
            2,
            2,
            1,
            2,
            "avg",
            "half_pixel_for_nn",
            "roi_align_v9_avg_half_pixel_for_nn"),

        ROIAlignV9Params(
            PartialShape{2, 1, 8, 8},
            ET,
            std::vector<T>{0, 1, 8, 5, 5,  2, 0,  7, 7, 10, 4,  5, 9,  0, 0, 5,  7,  0, 4, 0, 4, 7,  6,  10, 9,  5,
                           1, 7, 4, 7, 10, 8, 2,  0, 8, 3,  6,  8, 10, 4, 2, 10, 7,  8, 7, 0, 6, 9,  2,  4,  8,  5,
                           2, 3, 3, 1, 5,  9, 10, 0, 9, 5,  5,  3, 10, 5, 2, 0,  10, 0, 5, 4, 3, 10, 5,  5,  10, 0,
                           8, 8, 9, 1, 0,  7, 9,  6, 8, 7,  10, 9, 2,  3, 3, 5,  6,  9, 4, 9, 2, 4,  5,  5,  3,  1,
                           1, 6, 8, 0, 5,  5, 10, 8, 6, 9,  6,  9, 1,  2, 7, 1,  1,  3, 0, 4, 0, 7,  10, 2},
            reference_tests::Tensor(ET,
                                    {2, 1, 2, 2},
                                    std::vector<T>{4.375, 4.9375, 5.6875, 5.625, 4.625, 7.125, 3.3125, 4.3125}),
            reference_tests::Tensor(ET, {2, 4}, std::vector<T>{2, 2, 4, 4, 2, 2, 4, 4}),
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{0, 1}),
            2,
            2,
            1,
            2,
            "max",
            "half_pixel",
            "roi_align_v9_max_half_pixel"),
    };
    return params;
}

std::vector<ROIAlignV9Params> generateCombinedParamsV9() {
    const std::vector<std::vector<ROIAlignV9Params>> generatedParams{
        generateParamsV9<element::Type_t::bf16, element::Type_t::i8>(),
        generateParamsV9<element::Type_t::f16, element::Type_t::i8>(),
        generateParamsV9<element::Type_t::f32, element::Type_t::i8>(),
        generateParamsV9<element::Type_t::bf16, element::Type_t::i16>(),
        generateParamsV9<element::Type_t::f16, element::Type_t::i16>(),
        generateParamsV9<element::Type_t::f32, element::Type_t::i16>(),
        generateParamsV9<element::Type_t::bf16, element::Type_t::i32>(),
        generateParamsV9<element::Type_t::f16, element::Type_t::i32>(),
        generateParamsV9<element::Type_t::f32, element::Type_t::i32>(),
        generateParamsV9<element::Type_t::bf16, element::Type_t::i64>(),
        generateParamsV9<element::Type_t::f16, element::Type_t::i64>(),
        generateParamsV9<element::Type_t::f32, element::Type_t::i64>(),
        generateParamsV9<element::Type_t::bf16, element::Type_t::u8>(),
        generateParamsV9<element::Type_t::f16, element::Type_t::u8>(),
        generateParamsV9<element::Type_t::f32, element::Type_t::u8>(),
        generateParamsV9<element::Type_t::bf16, element::Type_t::u16>(),
        generateParamsV9<element::Type_t::f16, element::Type_t::u16>(),
        generateParamsV9<element::Type_t::f32, element::Type_t::u16>(),
        generateParamsV9<element::Type_t::bf16, element::Type_t::u32>(),
        generateParamsV9<element::Type_t::f16, element::Type_t::u32>(),
        generateParamsV9<element::Type_t::f32, element::Type_t::u32>(),
        generateParamsV9<element::Type_t::bf16, element::Type_t::u64>(),
        generateParamsV9<element::Type_t::f16, element::Type_t::u64>(),
        generateParamsV9<element::Type_t::f32, element::Type_t::u64>(),
    };
    std::vector<ROIAlignV9Params> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ROIAlign_With_Hardcoded_Refs,
                         ReferenceROIAlignTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceROIAlignTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ROIAlignV9_With_Hardcoded_Refs,
                         ReferenceROIAlignV9Test,
                         testing::ValuesIn(generateCombinedParamsV9()),
                         ReferenceROIAlignV9Test::getTestCaseName);
}  // namespace
