// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset1.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ROIAlignParams {
    template <class IT>
    ROIAlignParams(const PartialShape& pShape, const element::Type& iType, const std::vector<IT>& iValues,
                   const reference_tests::Tensor& expectedFeatureMap,
                   const reference_tests::Tensor& coords, const reference_tests::Tensor& roiIdx,
                   const int32_t pooledH, const int32_t pooledW,
                   const float spatialScale, const int32_t poolingRatio, const std::string& poolingMode,
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
        const auto featureMap = std::make_shared<opset1::Parameter>(params.iType, params.pShape);
        const auto coords = std::make_shared<opset1::Constant>(params.coords.type, params.coords.shape, params.coords.data.data());
        const auto roisIdx = std::make_shared<opset1::Constant>(params.roiIdx.type, params.roiIdx.shape, params.roiIdx.data.data());
        const auto roi_align = std::make_shared<opset3::ROIAlign>(featureMap,
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

TEST_P(ReferenceROIAlignTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_IND>
std::vector<ROIAlignParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<ROIAlignParams> params {
            ROIAlignParams(PartialShape{2, 1, 8, 8}, ET,
                           std::vector<T>{0,  1, 8, 5, 5,  2, 0,  7,  7,  10, 4,  5, 9,  0, 0,  5,
                                          7,  0, 4, 0, 4,  7, 6,  10, 9,  5,  1,  7, 4,  7, 10, 8,
                                          2,  0, 8, 3, 6,  8, 10, 4,  2,  10, 7,  8, 7,  0, 6,  9,
                                          2,  4, 8, 5, 2,  3, 3,  1,  5,  9,  10, 0, 9,  5, 5,  3,
                                          10, 5, 2, 0, 10, 0, 5,  4,  3,  10, 5,  5, 10, 0, 8,  8,
                                          9,  1, 0, 7, 9,  6, 8,  7,  10, 9,  2,  3, 3,  5, 6,  9,
                                          4,  9, 2, 4, 5,  5, 3,  1,  1,  6,  8,  0, 5,  5, 10, 8,
                                          6,  9, 6, 9, 1,  2, 7,  1,  1,  3,  0,  4, 0,  7, 10, 2},
                           reference_tests::Tensor(ET, {2, 1, 2, 2}, std::vector<T>{3, 3.75, 4.75, 5, 3, 5.5, 2.75, 3.75}),
                           reference_tests::Tensor(ET, {2, 4}, std::vector<T>{2, 2, 4, 4, 2, 2, 4, 4}),
                           reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{0, 1}),
                           2, 2, 1, 2, "avg", "roi_align_avg"),

            ROIAlignParams(PartialShape{2, 1, 8, 8}, ET,
                           std::vector<T>{0,  1, 8, 5, 5,  2, 0,  7,  7,  10, 4,  5, 9,  0, 0,  5,
                                          7,  0, 4, 0, 4,  7, 6,  10, 9,  5,  1,  7, 4,  7, 10, 8,
                                          2,  0, 8, 3, 6,  8, 10, 4,  2,  10, 7,  8, 7,  0, 6,  9,
                                          2,  4, 8, 5, 2,  3, 3,  1,  5,  9,  10, 0, 9,  5, 5,  3,
                                          10, 5, 2, 0, 10, 0, 5,  4,  3,  10, 5,  5, 10, 0, 8,  8,
                                          9,  1, 0, 7, 9,  6, 8,  7,  10, 9,  2,  3, 3,  5, 6,  9,
                                          4,  9, 2, 4, 5,  5, 3,  1,  1,  6,  8,  0, 5,  5, 10, 8,
                                          6,  9, 6, 9, 1,  2, 7,  1,  1,  3,  0,  4, 0,  7, 10, 2},
                           reference_tests::Tensor(ET, {2, 1, 2, 2}, std::vector<T>{4.375, 4.9375, 5.6875, 5.625, 4.625, 7.125, 3.3125, 4.3125}),
                           reference_tests::Tensor(ET, {2, 4}, std::vector<T>{2, 2, 4, 4, 2, 2, 4, 4}),
                           reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{0, 1}),
                           2, 2, 1, 2, "max", "roi_align_max"),
    };
    return params;
}

std::vector<ROIAlignParams> generateCombinedParams() {
    const std::vector<std::vector<ROIAlignParams>> generatedParams {
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

INSTANTIATE_TEST_SUITE_P(smoke_ROIAlign_With_Hardcoded_Refs, ReferenceROIAlignTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceROIAlignTest::getTestCaseName);
} // namespace
