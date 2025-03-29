// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/strided_slice.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct StridedSliceParams {
    StridedSliceParams(const PartialShape& dynamicDataShape,
                       const reference_tests::Tensor& dataTensor,
                       const reference_tests::Tensor& beginTensor,
                       const reference_tests::Tensor& endTensor,
                       const reference_tests::Tensor& stridesTensor,
                       const std::vector<int64_t>& beginMask,
                       const std::vector<int64_t>& endMask,
                       const std::vector<int64_t>& newAxisMask,
                       const std::vector<int64_t>& shrinkAxisMask,
                       const std::vector<int64_t>& ellipsisMask,
                       const reference_tests::Tensor& expectedTensor,
                       const std::string& testcaseName = "")
        : dynamicDataShape(dynamicDataShape),
          dataTensor(dataTensor),
          beginTensor(beginTensor),
          endTensor(endTensor),
          stridesTensor(stridesTensor),
          beginMask(beginMask),
          endMask(endMask),
          newAxisMask(newAxisMask),
          shrinkAxisMask(shrinkAxisMask),
          ellipsisMask(ellipsisMask),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    PartialShape dynamicDataShape;
    reference_tests::Tensor dataTensor;
    reference_tests::Tensor beginTensor;
    reference_tests::Tensor endTensor;
    reference_tests::Tensor stridesTensor;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> ellipsisMask;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

struct StridedSliceStrideOptionalParams {
    StridedSliceStrideOptionalParams(const PartialShape& dynamicDataShape,
                                     const reference_tests::Tensor& dataTensor,
                                     const reference_tests::Tensor& beginTensor,
                                     const reference_tests::Tensor& endTensor,
                                     const std::vector<int64_t>& beginMask,
                                     const std::vector<int64_t>& endMask,
                                     const std::vector<int64_t>& newAxisMask,
                                     const std::vector<int64_t>& shrinkAxisMask,
                                     const std::vector<int64_t>& ellipsisMask,
                                     const reference_tests::Tensor& expectedTensor,
                                     const std::string& testcaseName = "")
        : dynamicDataShape(dynamicDataShape),
          dataTensor(dataTensor),
          beginTensor(beginTensor),
          endTensor(endTensor),
          beginMask(beginMask),
          endMask(endMask),
          newAxisMask(newAxisMask),
          shrinkAxisMask(shrinkAxisMask),
          ellipsisMask(ellipsisMask),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    PartialShape dynamicDataShape;
    reference_tests::Tensor dataTensor;
    reference_tests::Tensor beginTensor;
    reference_tests::Tensor endTensor;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> ellipsisMask;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceStridedSliceLayerTest : public testing::TestWithParam<StridedSliceParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        if (params.dynamicDataShape.is_static()) {
            inputData = {params.dataTensor.data};
        } else {
            inputData = {params.dataTensor.data,
                         params.beginTensor.data,
                         params.endTensor.data,
                         params.stridesTensor.data};
        }
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "ddShape=" << param.dynamicDataShape;
        result << "_dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_bType=" << param.beginTensor.type;
        result << "_bShape=" << param.beginTensor.shape;
        result << "_eType=" << param.endTensor.type;
        result << "_eShape=" << param.endTensor.shape;
        result << "_sType=" << param.stridesTensor.type;
        result << "_sShape=" << param.stridesTensor.shape;
        result << "_eType=" << param.expectedTensor.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expectedTensor.shape;
            result << "_" << param.testcaseName;
        } else {
            result << "_eShape=" << param.expectedTensor.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const StridedSliceParams& params) {
        std::shared_ptr<Model> function;
        if (params.dynamicDataShape.is_static()) {
            const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
            const auto beginOp = std::make_shared<op::v0::Constant>(params.beginTensor.type,
                                                                    params.beginTensor.shape,
                                                                    params.beginTensor.data.data());
            const auto endOp = std::make_shared<op::v0::Constant>(params.endTensor.type,
                                                                  params.endTensor.shape,
                                                                  params.endTensor.data.data());
            const auto stridesOp = std::make_shared<op::v0::Constant>(params.stridesTensor.type,
                                                                      params.stridesTensor.shape,
                                                                      params.stridesTensor.data.data());
            const auto StridedSlice = std::make_shared<op::v1::StridedSlice>(data,
                                                                             beginOp,
                                                                             endOp,
                                                                             stridesOp,
                                                                             params.beginMask,
                                                                             params.endMask,
                                                                             params.newAxisMask,
                                                                             params.shrinkAxisMask,
                                                                             params.ellipsisMask);
            function = std::make_shared<ov::Model>(NodeVector{StridedSlice}, ParameterVector{data});
        } else {
            const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, PartialShape::dynamic());
            const auto beginOp = std::make_shared<op::v0::Parameter>(params.beginTensor.type, params.beginTensor.shape);
            const auto endOp = std::make_shared<op::v0::Parameter>(params.endTensor.type, params.endTensor.shape);
            const auto stridesOp =
                std::make_shared<op::v0::Parameter>(params.stridesTensor.type, params.stridesTensor.shape);
            const auto StridedSlice = std::make_shared<op::v1::StridedSlice>(data,
                                                                             beginOp,
                                                                             endOp,
                                                                             stridesOp,
                                                                             params.beginMask,
                                                                             params.endMask,
                                                                             params.newAxisMask,
                                                                             params.shrinkAxisMask,
                                                                             params.ellipsisMask);
            function =
                std::make_shared<ov::Model>(NodeVector{StridedSlice}, ParameterVector{data, beginOp, endOp, stridesOp});
        }
        return function;
    }
};

class ReferenceStridedSliceLayerTestStrideOptional : public testing::TestWithParam<StridedSliceStrideOptionalParams>,
                                                     public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        if (params.dynamicDataShape.is_static()) {
            inputData = {params.dataTensor.data};
        } else {
            inputData = {params.dataTensor.data, params.beginTensor.data, params.endTensor.data};
        }
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceStrideOptionalParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "ddShape=" << param.dynamicDataShape;
        result << "_dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_bType=" << param.beginTensor.type;
        result << "_bShape=" << param.beginTensor.shape;
        result << "_eType=" << param.endTensor.type;
        result << "_eShape=" << param.endTensor.shape;
        result << "_eType=" << param.expectedTensor.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expectedTensor.shape;
            result << "_" << param.testcaseName;
        } else {
            result << "_eShape=" << param.expectedTensor.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const StridedSliceStrideOptionalParams& params) {
        std::shared_ptr<Model> function;
        if (params.dynamicDataShape.is_static()) {
            const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
            const auto beginOp = std::make_shared<op::v0::Constant>(params.beginTensor.type,
                                                                    params.beginTensor.shape,
                                                                    params.beginTensor.data.data());
            const auto endOp = std::make_shared<op::v0::Constant>(params.endTensor.type,
                                                                  params.endTensor.shape,
                                                                  params.endTensor.data.data());
            const auto StridedSlice = std::make_shared<op::v1::StridedSlice>(data,
                                                                             beginOp,
                                                                             endOp,
                                                                             params.beginMask,
                                                                             params.endMask,
                                                                             params.newAxisMask,
                                                                             params.shrinkAxisMask,
                                                                             params.ellipsisMask);
            function = std::make_shared<ov::Model>(NodeVector{StridedSlice}, ParameterVector{data});
        } else {
            const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, PartialShape::dynamic());
            const auto beginOp = std::make_shared<op::v0::Parameter>(params.beginTensor.type, params.beginTensor.shape);
            const auto endOp = std::make_shared<op::v0::Parameter>(params.endTensor.type, params.endTensor.shape);
            const auto StridedSlice = std::make_shared<op::v1::StridedSlice>(data,
                                                                             beginOp,
                                                                             endOp,
                                                                             params.beginMask,
                                                                             params.endMask,
                                                                             params.newAxisMask,
                                                                             params.shrinkAxisMask,
                                                                             params.ellipsisMask);
            function = std::make_shared<ov::Model>(NodeVector{StridedSlice}, ParameterVector{data, beginOp, endOp});
        }
        return function;
    }
};

TEST_P(ReferenceStridedSliceLayerTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceStridedSliceLayerTestStrideOptional, CompareWithRefs) {
    Exec();
}

template <typename T>
std::vector<T> generateInputValues(const Shape& input_shape, T initial) {
    std::vector<T> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), static_cast<T>(initial));
    return input_values;
}

template <element::Type_t IN_ET>
std::vector<StridedSliceParams> generateSmallParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<StridedSliceParams> params{
        // strided_slice_0
        StridedSliceParams(
            {},
            reference_tests::Tensor(IN_ET, {2, 3, 4}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{1, 0}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{0, 0}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{1, 1}),
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 1, 0},
            std::vector<int64_t>{1, 0, 0},
            std::vector<int64_t>{0, 0, 0},
            reference_tests::Tensor(IN_ET, {1, 3, 4}, std::vector<T>{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            "strided_slice_0"),
        // strided_slice_0_dynamic
        StridedSliceParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {2, 3, 4}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{1, 0}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{0, 0}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{1, 1}),
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 1, 0},
            std::vector<int64_t>{1, 0, 0},
            std::vector<int64_t>{0, 0, 0},
            reference_tests::Tensor(IN_ET, {1, 3, 4}, std::vector<T>{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            "strided_slice_0_dynamic"),
    };
    return params;
}

template <element::Type_t IN_ET>
std::vector<StridedSliceParams> generateParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<StridedSliceParams> params{
        // strided_slice_0
        StridedSliceParams(
            {},
            reference_tests::Tensor(IN_ET, {2, 3, 4}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{1, 0}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{0, 0}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{1, 1}),
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 1, 0},
            std::vector<int64_t>{1, 0, 0},
            std::vector<int64_t>{0, 0, 0},
            reference_tests::Tensor(IN_ET, {1, 3, 4}, std::vector<T>{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            "strided_slice_0"),
        // strided_slice_0_dynamic
        StridedSliceParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {2, 3, 4}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{1, 0}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{0, 0}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{1, 1}),
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 1, 0},
            std::vector<int64_t>{1, 0, 0},
            std::vector<int64_t>{0, 0, 0},
            reference_tests::Tensor(IN_ET, {1, 3, 4}, std::vector<T>{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            "strided_slice_0_dynamic"),
        // strided_slice_1
        StridedSliceParams(
            {},
            reference_tests::Tensor(IN_ET, {2, 4, 6, 8, 2, 2, 2}, generateInputValues<T>({2, 4, 6, 8, 2, 2, 2}, 0)),
            reference_tests::Tensor(element::i64, {7}, std::vector<int64_t>{0, 0, 2, 7, 0, 0, 1}),
            reference_tests::Tensor(element::i64, {7}, std::vector<int64_t>{0, 4, 6, 3, 0, 0, 0}),
            reference_tests::Tensor(element::i64, {7}, std::vector<int64_t>{1, 1, 2, -2, 1, 1, 1}),
            std::vector<int64_t>{0, 1, 0, 0, 0, 0, 0},
            std::vector<int64_t>{1, 0, 0, 0, 0, 0, 0},
            std::vector<int64_t>{0, 0, 0, 0, 1, 0, 0},
            std::vector<int64_t>{0, 0, 0, 0, 0, 0, 1},
            std::vector<int64_t>{0, 0, 0, 0, 0, 1, 0},
            reference_tests::Tensor(
                IN_ET,
                {2, 4, 2, 2, 1, 2, 2},
                std::vector<T>{185,  187,  189,  191,  169,  171,  173,  175,  313,  315,  317,  319,  297,  299,  301,
                               303,  569,  571,  573,  575,  553,  555,  557,  559,  697,  699,  701,  703,  681,  683,
                               685,  687,  953,  955,  957,  959,  937,  939,  941,  943,  1081, 1083, 1085, 1087, 1065,
                               1067, 1069, 1071, 1337, 1339, 1341, 1343, 1321, 1323, 1325, 1327, 1465, 1467, 1469, 1471,
                               1449, 1451, 1453, 1455, 1721, 1723, 1725, 1727, 1705, 1707, 1709, 1711, 1849, 1851, 1853,
                               1855, 1833, 1835, 1837, 1839, 2105, 2107, 2109, 2111, 2089, 2091, 2093, 2095, 2233, 2235,
                               2237, 2239, 2217, 2219, 2221, 2223, 2489, 2491, 2493, 2495, 2473, 2475, 2477, 2479, 2617,
                               2619, 2621, 2623, 2601, 2603, 2605, 2607, 2873, 2875, 2877, 2879, 2857, 2859, 2861, 2863,
                               3001, 3003, 3005, 3007, 2985, 2987, 2989, 2991}),
            "strided_slice_1"),
        // strided_slice_1_dynamic
        StridedSliceParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {2, 4, 6, 8, 2, 2, 2}, generateInputValues<T>({2, 4, 6, 8, 2, 2, 2}, 0)),
            reference_tests::Tensor(element::i64, {7}, std::vector<int64_t>{0, 0, 2, 7, 0, 0, 1}),
            reference_tests::Tensor(element::i64, {7}, std::vector<int64_t>{0, 4, 6, 3, 0, 0, 0}),
            reference_tests::Tensor(element::i64, {7}, std::vector<int64_t>{1, 1, 2, -2, 1, 1, 1}),
            std::vector<int64_t>{0, 1, 0, 0, 0, 0, 0},
            std::vector<int64_t>{1, 0, 0, 0, 0, 0, 0},
            std::vector<int64_t>{0, 0, 0, 0, 1, 0, 0},
            std::vector<int64_t>{0, 0, 0, 0, 0, 0, 1},
            std::vector<int64_t>{0, 0, 0, 0, 0, 1, 0},
            reference_tests::Tensor(
                IN_ET,
                {2, 4, 2, 2, 1, 2, 2},
                std::vector<T>{185,  187,  189,  191,  169,  171,  173,  175,  313,  315,  317,  319,  297,  299,  301,
                               303,  569,  571,  573,  575,  553,  555,  557,  559,  697,  699,  701,  703,  681,  683,
                               685,  687,  953,  955,  957,  959,  937,  939,  941,  943,  1081, 1083, 1085, 1087, 1065,
                               1067, 1069, 1071, 1337, 1339, 1341, 1343, 1321, 1323, 1325, 1327, 1465, 1467, 1469, 1471,
                               1449, 1451, 1453, 1455, 1721, 1723, 1725, 1727, 1705, 1707, 1709, 1711, 1849, 1851, 1853,
                               1855, 1833, 1835, 1837, 1839, 2105, 2107, 2109, 2111, 2089, 2091, 2093, 2095, 2233, 2235,
                               2237, 2239, 2217, 2219, 2221, 2223, 2489, 2491, 2493, 2495, 2473, 2475, 2477, 2479, 2617,
                               2619, 2621, 2623, 2601, 2603, 2605, 2607, 2873, 2875, 2877, 2879, 2857, 2859, 2861, 2863,
                               3001, 3003, 3005, 3007, 2985, 2987, 2989, 2991}),
            "strided_slice_1_dynamic"),
    };
    return params;
}

template <element::Type_t IN_ET>
std::vector<StridedSliceStrideOptionalParams> generateStrideOptionalParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<StridedSliceStrideOptionalParams> params{
        // strided_slice_stride_optional
        StridedSliceStrideOptionalParams(
            {},
            reference_tests::Tensor(IN_ET, {2, 3, 4}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i64, {3}, std::vector<int64_t>{-1, -1, 0}),
            reference_tests::Tensor(element::i64, {3}, std::vector<int64_t>{0, 0, 0}),
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 0, 1},
            std::vector<int64_t>{1, 1, 0},
            std::vector<int64_t>{0, 0, 0},
            reference_tests::Tensor(IN_ET, {1, 4}, std::vector<T>{20, 21, 22, 23}),
            "strided_slice_stride_optional"),
        // strided_slice_stride_optional_dynamic
        StridedSliceStrideOptionalParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {2, 3, 4}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i64, {3}, std::vector<int64_t>{-1, -1, 0}),
            reference_tests::Tensor(element::i64, {3}, std::vector<int64_t>{0, 0, 0}),
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{0, 0, 1},
            std::vector<int64_t>{1, 1, 0},
            std::vector<int64_t>{0, 0, 0},
            reference_tests::Tensor(IN_ET, {1, 4}, std::vector<T>{20, 21, 22, 23}),
            "strided_slice_stride_optional_dynamic"),
        // strided_slice_stride_optional_dynamic_empty_output_tensor
        StridedSliceStrideOptionalParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {2, 3, 4}, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{0, 0}),
            reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{-1, 0}),
            std::vector<int64_t>{1, 0},
            std::vector<int64_t>{1, 0},
            std::vector<int64_t>{},
            std::vector<int64_t>{},
            std::vector<int64_t>{},
            reference_tests::Tensor(IN_ET, {2, 0, 4}, std::vector<T>{}),
            "strided_slice_stride_optional_dynamic_empty_output_tensor"),
    };
    return params;
}

std::vector<StridedSliceParams> generateCombinedParams() {
    const std::vector<std::vector<StridedSliceParams>> generatedParams{
        generateSmallParams<element::Type_t::i8>(),
        generateParams<element::Type_t::i16>(),
        generateParams<element::Type_t::i32>(),
        generateParams<element::Type_t::i64>(),
        generateSmallParams<element::Type_t::u8>(),
        generateParams<element::Type_t::u16>(),
        generateParams<element::Type_t::u32>(),
        generateParams<element::Type_t::u64>(),
        generateSmallParams<element::Type_t::bf16>(),
        generateSmallParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<StridedSliceParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

std::vector<StridedSliceStrideOptionalParams> generateCombinedStrideOptionalParams() {
    const std::vector<std::vector<StridedSliceStrideOptionalParams>> generatedParams{
        generateStrideOptionalParams<element::Type_t::i8>(),
        generateStrideOptionalParams<element::Type_t::i16>(),
        generateStrideOptionalParams<element::Type_t::i32>(),
        generateStrideOptionalParams<element::Type_t::i64>(),
        generateStrideOptionalParams<element::Type_t::u8>(),
        generateStrideOptionalParams<element::Type_t::u16>(),
        generateStrideOptionalParams<element::Type_t::u32>(),
        generateStrideOptionalParams<element::Type_t::u64>(),
        generateStrideOptionalParams<element::Type_t::f32>(),
        generateStrideOptionalParams<element::Type_t::f64>(),
    };
    std::vector<StridedSliceStrideOptionalParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_StridedSlice_With_Hardcoded_Refs,
                         ReferenceStridedSliceLayerTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceStridedSliceLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StridedSlice_With_Hardcoded_Refs,
                         ReferenceStridedSliceLayerTestStrideOptional,
                         testing::ValuesIn(generateCombinedStrideOptionalParams()),
                         ReferenceStridedSliceLayerTestStrideOptional::getTestCaseName);
}  // namespace
