// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/col2im.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/constant.hpp"

namespace {
struct Col2ImParams {
    Col2ImParams(const reference_tests::Tensor& dataTensor,
                 const reference_tests::Tensor& outputSizeTensor,
                 const reference_tests::Tensor& kernelSizeTensor,
                 const reference_tests::Tensor& expectedTensor,
                 const ov::Strides& strides,
                 const ov::Strides& dilations,
                 const ov::Shape& pads_begin,
                 const ov::Shape& pads_end)
        : dataTensor(dataTensor),
          outputSizeTensor(outputSizeTensor),
          kernelSizeTensor(kernelSizeTensor),
          expectedTensor(expectedTensor),
          strides(strides),
          dilations(dilations),
          pads_begin(pads_begin),
          pads_end(pads_end) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor outputSizeTensor;
    reference_tests::Tensor kernelSizeTensor;
    reference_tests::Tensor expectedTensor;
    ov::Strides strides;
    ov::Strides dilations;
    ov::Shape pads_begin;
    ov::Shape pads_end;
};

class ReferenceCol2ImV15LayerTest : public testing::TestWithParam<Col2ImParams>,
                                    public reference_tests::CommonReferenceTest {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data, params.outputSizeTensor.data, params.kernelSizeTensor.data};
        refOutData = {params.expectedTensor.data};
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<Col2ImParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_outputSizeType=" << param.outputSizeTensor.type;
        result << "_kernelSizeType=" << param.kernelSizeTensor.type;
        result << "_strides=" << param.strides;
        result << "_dilations=" << param.dilations;
        result << "_pads_begin=" << param.pads_begin;
        result << "_pads_end=" << param.pads_end;
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const Col2ImParams& params) {
        using ov::op::v0::Parameter;
        const auto data = std::make_shared<Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto output_size =
            std::make_shared<Parameter>(params.outputSizeTensor.type, params.outputSizeTensor.shape);
        const auto kernel_size =
            std::make_shared<Parameter>(params.kernelSizeTensor.type, params.kernelSizeTensor.shape);
        const auto col2im = std::make_shared<ov::op::v15::Col2Im>(data,
                                                                  output_size,
                                                                  kernel_size,
                                                                  params.strides,
                                                                  params.dilations,
                                                                  params.pads_begin,
                                                                  params.pads_end);
        return std::make_shared<ov::Model>(ov::NodeVector{col2im}, ov::ParameterVector{data, output_size, kernel_size});
    }
};

TEST_P(ReferenceCol2ImV15LayerTest, CompareWithRefs) {
    Exec();
}

template <typename T>
std::vector<T> getContinuousIncreasingValue(size_t elementSize, float step) {
    std::vector<T> a(elementSize);
    std::iota(std::begin(a), std::end(a), step);
    return a;
}

template <ov::element::Type_t T, ov::element::Type_t T_idx>
std::vector<Col2ImParams> generateCol2ImParams() {
    using T_D = typename ov::element_type_traits<T>::value_type;
    using T_I = typename ov::element_type_traits<T_idx>::value_type;
    using reference_tests::Tensor;
    std::vector<Col2ImParams> col2ImParams{
        // non-batched, no pads, default arguments, 1x1 kernel_size
        Col2ImParams(Tensor({3, 4}, T, getContinuousIncreasingValue<T_D>(12, 1)),
                     Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
                     Tensor({2}, T_idx, std::vector<T_I>{1, 1}),
                     Tensor({3, 2, 2}, T, getContinuousIncreasingValue<T_D>(12, 1)),
                     {1, 1},
                     {1, 1},
                     {0, 0},
                     {0, 0}),
        // non-batched, no pads, default strides and dilations, 2x2 kernel_size
        Col2ImParams(Tensor({12, 9}, T, getContinuousIncreasingValue<T_D>(108, 1)),
                     Tensor({2}, T_idx, std::vector<T_I>{4, 4}),
                     Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
                     Tensor({3, 4, 4}, T, std::vector<T_D>{1,   12,  14,  12,  23,  66,  70,  45,  29, 78,  82,  51,
                                                           25,  60,  62,  36,  37,  84,  86,  48,  95, 210, 214, 117,
                                                           101, 222, 226, 123, 61,  132, 134, 72,  73, 156, 158, 84,
                                                           167, 354, 358, 189, 173, 366, 370, 195, 97, 204, 206, 108}),
                     {1, 1},
                     {1, 1},
                     {0, 0},
                     {0, 0}),
        // non-batched, (1, 1) pads, default strides and dilations, 2x2 kernel_size
        Col2ImParams(
            Tensor({12, 25}, T, getContinuousIncreasingValue<T_D>(300, 1)),
            Tensor({2}, T_idx, std::vector<T_I>{4, 4}),
            Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
            Tensor({3, 4, 4}, T, std::vector<T_D>{166, 170, 174, 178, 186,  190,  194,  198,  206,  210,  214,  218,
                                                  226, 230, 234, 238, 566,  570,  574,  578,  586,  590,  594,  598,
                                                  606, 610, 614, 618, 626,  630,  634,  638,  966,  970,  974,  978,
                                                  986, 990, 994, 998, 1006, 1010, 1014, 1018, 1026, 1030, 1034, 1038}),
            {1, 1},
            {1, 1},
            {1, 1},
            {1, 1}),
        // non-batched, (3, 3) pads, non-default strides, default dilations, 2x2 kernel_size
        Col2ImParams(Tensor({12, 25}, T, getContinuousIncreasingValue<T_D>(300, 1)),
                     Tensor({2}, T_idx, std::vector<T_I>{4, 4}),
                     Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
                     Tensor({3, 4, 4}, T, std::vector<T_D>{82,  58,  83,  59,  37,  13,  38,  14,  87,  63,  88,  64,
                                                           42,  18,  43,  19,  182, 158, 183, 159, 137, 113, 138, 114,
                                                           187, 163, 188, 164, 142, 118, 143, 119, 282, 258, 283, 259,
                                                           237, 213, 238, 214, 287, 263, 288, 264, 242, 218, 243, 219}),
                     {2, 2},
                     {1, 1},
                     {3, 3},
                     {3, 3}),
        // non-batched, 4 channels, (0, 2) pads, non-default strides, non-default dilations, 3x3 kernel_size
        Col2ImParams(
            Tensor({36, 3}, T, getContinuousIncreasingValue<T_D>(108, 1)),
            Tensor({2}, T_idx, std::vector<T_I>{5, 5}),
            Tensor({2}, T_idx, std::vector<T_I>{3, 3}),
            Tensor({4, 5, 5},
                   T,
                   std::vector<T_D>{
                       6,   0, 15,  0, 14,  0, 0, 0, 0, 0, 24,  0, 42,  0, 32,  0, 0, 0, 0, 0, 42,  0, 69,  0, 50,
                       60,  0, 96,  0, 68,  0, 0, 0, 0, 0, 78,  0, 123, 0, 86,  0, 0, 0, 0, 0, 96,  0, 150, 0, 104,
                       114, 0, 177, 0, 122, 0, 0, 0, 0, 0, 132, 0, 204, 0, 140, 0, 0, 0, 0, 0, 150, 0, 231, 0, 158,
                       168, 0, 258, 0, 176, 0, 0, 0, 0, 0, 186, 0, 285, 0, 194, 0, 0, 0, 0, 0, 204, 0, 312, 0, 212}),
            {2, 2},
            {2, 2},
            {0, 2},
            {0, 2}),
        // batched, non-default pads, default strides and dilations, 2x2 kernel_size
        Col2ImParams(
            Tensor({2, 12, 9}, T, getContinuousIncreasingValue<T_D>(216, 1)),
            Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
            Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
            Tensor({2, 3, 2, 2}, T, std::vector<T_D>{66,  70,  78,  82,  210, 214, 222, 226, 354, 358, 366, 370,
                                                     498, 502, 510, 514, 642, 646, 654, 658, 786, 790, 798, 802}),
            {1, 1},
            {1, 1},
            {1, 1},
            {1, 1}),
        // batched, 2 channels, (0, 2) pads, non-default strides, default dilations, 2x2 kernel_size
        Col2ImParams(
            Tensor({2, 8, 8}, T, getContinuousIncreasingValue<T_D>(128, 1)),
            Tensor({2}, T_idx, std::vector<T_I>{5, 5}),
            Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
            Tensor({2, 2, 5, 5},
                   T,
                   std::vector<T_D>{2,   10,  3,   11,  4,   18,  26,  19,  27,  20,  6,   14,  7,   15,  8,   22,  30,
                                    23,  31,  24,  0,   0,   0,   0,   0,   34,  42,  35,  43,  36,  50,  58,  51,  59,
                                    52,  38,  46,  39,  47,  40,  54,  62,  55,  63,  56,  0,   0,   0,   0,   0,   66,
                                    74,  67,  75,  68,  82,  90,  83,  91,  84,  70,  78,  71,  79,  72,  86,  94,  87,
                                    95,  88,  0,   0,   0,   0,   0,   98,  106, 99,  107, 100, 114, 122, 115, 123, 116,
                                    102, 110, 103, 111, 104, 118, 126, 119, 127, 120, 0,   0,   0,   0,   0}),
            {2, 2},
            {1, 1},
            {0, 2},
            {0, 2}),
        // batched, 3 channels, (1, 1) pads, default strides, non-default dilations, 2x2 kernel_size
        Col2ImParams(
            Tensor({1, 12, 25}, T, getContinuousIncreasingValue<T_D>(300, 1)),
            Tensor({2}, T_idx, std::vector<T_I>{5, 5}),
            Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
            Tensor({1, 3, 5, 5},
                   T,
                   std::vector<T_D>{7,   39,  41,   43,   34,  64,  178,  182,  186,  118, 74,  198, 202, 206, 128,
                                    84,  218, 222,  226,  138, 67,  159,  161,  163,  94,  107, 239, 241, 243, 134,
                                    264, 578, 582,  586,  318, 274, 598,  602,  606,  328, 284, 618, 622, 626, 338,
                                    167, 359, 361,  363,  194, 207, 439,  441,  443,  234, 464, 978, 982, 986, 518,
                                    474, 998, 1002, 1006, 528, 484, 1018, 1022, 1026, 538, 267, 559, 561, 563, 294}),
            {1, 1},
            {2, 2},
            {1, 1},
            {1, 1}),
        // batched, 1 channel, (1, 0) pads, non-default strides, non-default dilations, 2x2 kernel_size
        Col2ImParams(Tensor({1, 4, 3}, T, getContinuousIncreasingValue<T_D>(12, 1)),
                     Tensor({2}, T_idx, std::vector<T_I>{4, 4}),
                     Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
                     Tensor({1, 1, 4, 4}, T, std::vector<T_D>{9, 0, 15, 0, 0, 0, 0, 0, 11, 0, 17, 0, 0, 0, 0, 0}),
                     {2, 2},
                     {2, 2},
                     {2, 0},
                     {2, 0}),
        // batched, 2 channels, (4, 3) pads, non-default strides, non-default dilations, 4x4 kernel_size
        Col2ImParams(
            Tensor({2, 32, 12}, T, getContinuousIncreasingValue<T_D>(768, 1)),
            Tensor({2}, T_idx, std::vector<T_I>{6, 6}),
            Tensor({2}, T_idx, std::vector<T_I>{4, 4}),
            Tensor({2, 2, 6, 6},
                   T,
                   std::vector<T_D>{
                       0, 585,  0, 693,  0, 501,  0, 0, 0, 0, 0, 0, 0, 1086, 0, 1230, 0, 872,  0, 0, 0, 0, 0, 0,
                       0, 1044, 0, 1152, 0, 807,  0, 0, 0, 0, 0, 0, 0, 2313, 0, 2421, 0, 1653, 0, 0, 0, 0, 0, 0,
                       0, 3390, 0, 3534, 0, 2408, 0, 0, 0, 0, 0, 0, 0, 2772, 0, 2880, 0, 1959, 0, 0, 0, 0, 0, 0,
                       0, 4041, 0, 4149, 0, 2805, 0, 0, 0, 0, 0, 0, 0, 5694, 0, 5838, 0, 3944, 0, 0, 0, 0, 0, 0,
                       0, 4500, 0, 4608, 0, 3111, 0, 0, 0, 0, 0, 0, 0, 5769, 0, 5877, 0, 3957, 0, 0, 0, 0, 0, 0,
                       0, 7998, 0, 8142, 0, 5480, 0, 0, 0, 0, 0, 0, 0, 6228, 0, 6336, 0, 4263, 0, 0, 0, 0, 0, 0}),
            {2, 2},
            {2, 2},
            {4, 3},
            {4, 3}),
    };
    return col2ImParams;
}

std::vector<Col2ImParams> generateCol2ImV15CombinedParams() {
    using ov::element::Type_t;
    const std::vector<std::vector<Col2ImParams>> col2ImTypeParams{
        generateCol2ImParams<Type_t::i32, Type_t::i32>(),
        generateCol2ImParams<Type_t::i64, Type_t::i32>(),
        generateCol2ImParams<Type_t::u32, Type_t::i32>(),
        generateCol2ImParams<Type_t::u64, Type_t::i32>(),
        generateCol2ImParams<Type_t::f16, Type_t::i32>(),
        generateCol2ImParams<Type_t::f32, Type_t::i32>(),
        generateCol2ImParams<Type_t::i32, Type_t::i64>(),
        generateCol2ImParams<Type_t::i64, Type_t::i64>(),
        generateCol2ImParams<Type_t::u32, Type_t::i64>(),
        generateCol2ImParams<Type_t::u64, Type_t::i64>(),
        generateCol2ImParams<Type_t::f16, Type_t::i64>(),
        generateCol2ImParams<Type_t::f32, Type_t::i64>(),
    };

    std::vector<Col2ImParams> combinedParams;
    for (const auto& params : col2ImTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Col2Im_With_Hardcoded_Refs,
                         ReferenceCol2ImV15LayerTest,
                         testing::ValuesIn(generateCol2ImV15CombinedParams()),
                         ReferenceCol2ImV15LayerTest::getTestCaseName);

}  // namespace
