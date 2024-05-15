// Copyright (C) 2018-2024 Intel Corporation
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
                 const ov::Shape& pads_end,
                 const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          outputSizeTensor(outputSizeTensor),
          kernelSizeTensor(kernelSizeTensor),
          expectedTensor(expectedTensor),
          strides(strides),
          dilations(dilations),
          pads_begin(pads_begin),
          pads_end(pads_end),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor outputSizeTensor;
    reference_tests::Tensor kernelSizeTensor;
    reference_tests::Tensor expectedTensor;
    ov::Strides strides;
    ov::Strides dilations;
    ov::Shape pads_begin;
    ov::Shape pads_end;
    std::string testcaseName;
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
        // test_name
        Col2ImParams(Tensor({3, 4}, T, getContinuousIncreasingValue<T_D>(12, 1)),
                     Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
                     Tensor({2}, T_idx, std::vector<T_I>{1, 1}),
                     Tensor({3, 2, 2}, T, getContinuousIncreasingValue<T_D>(12, 1)),
                     {1, 1},
                     {1, 1},
                     {0, 0},
                     {0, 0},
                     "first_test"),
        // kernel_size
        Col2ImParams(Tensor({12, 9}, T, getContinuousIncreasingValue<T_D>(12*9, 1)),
                     Tensor({2}, T_idx, std::vector<T_I>{4, 4}),
                     Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
                     Tensor({3, 4, 4}, T, std::vector<T_D>{1,  12,  14,  12,  23,  66,  70,  45,  29,  78,  82,  51,
         25,  60,  62,  36,  37,  84,  86,  48,  95, 210, 214, 117,
        101, 222, 226, 123,  61, 132, 134,  72,  73, 156, 158,  84,
        167, 354, 358, 189, 173, 366, 370, 195,  97, 204, 206, 108}),
                     {1, 1},
                     {1, 1},
                     {0, 0},
                     {0, 0},
                     "kernel_size"),
        //Col2ImParams(Tensor({12, 25}, T, getContinuousIncreasingValue<T_D>(300, 1)),
        //             Tensor({2}, T_idx, std::vector<T_I>{4, 4}),
        //             Tensor({2}, T_idx, std::vector<T_I>{2, 2}),
        //             Tensor({3, 4, 4}, T, std::vector<T_D>{
        // 166,  170,  174,  178,  186,  190,  194,  198,  206,  210,
        // 214,  218,  226,  230,  234,  238,  566,  570,  574,  578,
        // 586,  590,  594,  598,  606,  610,  614,  618,  626,  630,
        // 634,  638,  966,  970,  974,  978,  986,  990,  994,  998,
        // 1006, 1010, 1014, 1018, 1026, 1030, 1034, 1038}),
        //             {1, 1},
        //             {1, 1},
        //             {1, 1},
        //             {1, 1},
        //             "first_teste"),
    };
    return col2ImParams;
}

std::vector<Col2ImParams> generateCol2ImV15CombinedParams() {
    using ov::element::Type_t;
    const std::vector<std::vector<Col2ImParams>> col2ImTypeParams{
        generateCol2ImParams<Type_t::i32, Type_t::i32>(),
        //generateCol2ImParams<Type_t::i64, Type_t::i32>(),   // :(
        //generateCol2ImParams<Type_t::u32, Type_t::i32>(),
        //generateCol2ImParams<Type_t::u64, Type_t::i32>(),   // :(
        //generateCol2ImParams<Type_t::f16, Type_t::i32>(),   // :(
        //generateCol2ImParams<Type_t::f32, Type_t::i32>(),   // :(
        //generateCol2ImParams<Type_t::i32, Type_t::i64>(),   // :(
        //generateCol2ImParams<Type_t::i64, Type_t::i64>(),
        //generateCol2ImParams<Type_t::u32, Type_t::i64>(),   // :(
        //generateCol2ImParams<Type_t::u64, Type_t::i64>(),
        //generateCol2ImParams<Type_t::f16, Type_t::i64>(),   // :(
        //generateCol2ImParams<Type_t::f32, Type_t::i64>(),   // :(
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
