// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/one_hot.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct OneHotParams {
    OneHotParams(const reference_tests::Tensor& dataTensor,
                 const int32_t axis,
                 const reference_tests::Tensor& depthTensor,
                 const reference_tests::Tensor& onValueTensor,
                 const reference_tests::Tensor& offValueTensor,
                 const reference_tests::Tensor& expectedTensor,
                 const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          axis(axis),
          depthTensor(depthTensor),
          onValueTensor(onValueTensor),
          offValueTensor(offValueTensor),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    int32_t axis;
    reference_tests::Tensor depthTensor;
    reference_tests::Tensor onValueTensor;
    reference_tests::Tensor offValueTensor;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceOneHotTest : public testing::TestWithParam<OneHotParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<OneHotParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_axis=" << param.axis;
        result << "_deType=" << param.depthTensor.type;
        result << "_deShape=" << param.depthTensor.shape;
        result << "_onType=" << param.onValueTensor.type;
        result << "_onShape=" << param.onValueTensor.shape;
        result << "_offType=" << param.offValueTensor.type;
        result << "_offShape=" << param.offValueTensor.shape;
        result << "_eType=" << param.expectedTensor.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expectedTensor.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_eShape=" << param.expectedTensor.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const OneHotParams& params) {
        std::shared_ptr<Model> function;
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto depth = std::make_shared<op::v0::Constant>(params.depthTensor.type,
                                                              params.depthTensor.shape,
                                                              params.depthTensor.data.data());
        const auto onValue = std::make_shared<op::v0::Constant>(params.onValueTensor.type,
                                                                params.onValueTensor.shape,
                                                                params.onValueTensor.data.data());
        const auto offValue = std::make_shared<op::v0::Constant>(params.offValueTensor.type,
                                                                 params.offValueTensor.shape,
                                                                 params.offValueTensor.data.data());
        const auto oneHot = std::make_shared<op::v1::OneHot>(data, depth, onValue, offValue, params.axis);
        function = std::make_shared<ov::Model>(oneHot, ParameterVector{data});
        return function;
    }
};

TEST_P(ReferenceOneHotTest, CompareWithRefs) {
    Exec();
}

template <typename T>
std::vector<T> generateExpectedValues(const Shape& input_shape, std::vector<T> input, uint32_t category_count) {
    //    std::vector<T> input{0, 11, 101, 1001, 10001, static_cast<int32_t>(category_count - 1)};
    std::vector<T> output(shape_size(input_shape), 0);
    for (size_t i = 0; i < input.size(); ++i) {
        output[i * category_count + input[i]] = 1;
    }
    return output;
}

template <element::Type_t ET1, element::Type_t ET2>
std::vector<OneHotParams> generateParams() {
    using T1 = typename element_type_traits<ET1>::value_type;
    using T2 = typename element_type_traits<ET2>::value_type;
    std::vector<OneHotParams> params{
        OneHotParams(reference_tests::Tensor(ET1, {}, std::vector<T1>{2}),
                     0,
                     reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                     reference_tests::Tensor(ET2, {3}, std::vector<T2>{0, 0, 1}),
                     "one_hot_scalar_2_in_3"),
        OneHotParams(reference_tests::Tensor(ET1, {}, std::vector<T1>{1}),
                     0,
                     reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                     reference_tests::Tensor(ET2, {3}, std::vector<T2>{0, 1, 0}),
                     "one_hot_scalar_1_in_3"),
        OneHotParams(reference_tests::Tensor(ET1, {}, std::vector<T1>{0}),
                     0,
                     reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                     reference_tests::Tensor(ET2, {3}, std::vector<T2>{1, 0, 0}),
                     "one_hot_scalar_0_in_3"),
        OneHotParams(reference_tests::Tensor(ET1, {8}, std::vector<T1>{2, 1, 0, 0, 2, 2, 1, 0}),
                     0,
                     reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                     reference_tests::Tensor(ET2, {3, 8}, std::vector<T2>{0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
                                                                          0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0}),
                     "one_hot_vector_0"),
        OneHotParams(reference_tests::Tensor(ET1, {8}, std::vector<T1>{2, 1, 0, 0, 2, 2, 1, 0}),
                     1,
                     reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                     reference_tests::Tensor(ET2, {8, 3}, std::vector<T2>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                                                                          0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
                     "one_hot_vector_1"),
        OneHotParams(reference_tests::Tensor(ET1, {8}, std::vector<T1>{2, 1, 0, 0, 3, 2, 1, 0}),
                     1,
                     reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                     reference_tests::Tensor(ET2, {8, 3}, std::vector<T2>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                                                                          0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
                     "one_hot_vector_1_barely_oob"),
        OneHotParams(reference_tests::Tensor(ET1, {3, 3}, std::vector<T1>{0, 1, 1, 2, 1, 0, 0, 2, 1}),
                     0,
                     reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                     reference_tests::Tensor(ET2, {3, 3, 3}, std::vector<T2>{1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,
                                                                             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0}),
                     "one_hot_matrix_0"),
        OneHotParams(reference_tests::Tensor(ET1, {6}, std::vector<T1>{0, 11, 101, 1001, 10001, 19999}),
                     1,
                     reference_tests::Tensor(ET1, {}, std::vector<T1>{20000}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                     reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                     reference_tests::Tensor(
                         ET2,
                         {6, 20000},
                         generateExpectedValues({6, 20000}, std::vector<T2>{0, 11, 101, 1001, 10001, 19999}, 20000)),
                     "one_hot_vector_many_categories"),
    };
    return params;
}

template <element::Type_t ET1, element::Type_t ET2>
std::vector<OneHotParams> generateParamsFloat() {
    using T1 = typename element_type_traits<ET1>::value_type;
    using T2 = typename element_type_traits<ET2>::value_type;
    std::vector<OneHotParams> params{
        OneHotParams(
            reference_tests::Tensor(ET1, {3, 3}, std::vector<T1>{0, 1, 1, 2, 1, 0, 0, 2, 1}),
            0,
            reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
            reference_tests::Tensor(ET2, {}, std::vector<T2>{2.5}),
            reference_tests::Tensor(ET2, {}, std::vector<T2>{0.5}),
            reference_tests::Tensor(ET2, {3, 3, 3}, std::vector<T2>{2.5, 0.5, 0.5, 0.5, 0.5, 2.5, 2.5, 0.5, 0.5,
                                                                    0.5, 2.5, 2.5, 0.5, 2.5, 0.5, 0.5, 0.5, 2.5,
                                                                    0.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 2.5, 0.5}),
            "one_hot_on_off_float"),
    };
    return params;
}

std::vector<OneHotParams> generateCombinedParams() {
    const std::vector<std::vector<OneHotParams>> generatedParams{
        generateParams<element::Type_t::i32, element::Type_t::i16>(),
        generateParams<element::Type_t::i32, element::Type_t::i32>(),
        generateParams<element::Type_t::i32, element::Type_t::i64>(),
        generateParams<element::Type_t::i32, element::Type_t::u16>(),
        generateParams<element::Type_t::i32, element::Type_t::u32>(),
        generateParams<element::Type_t::i32, element::Type_t::u64>(),
        generateParams<element::Type_t::i64, element::Type_t::i16>(),
        generateParams<element::Type_t::i64, element::Type_t::i32>(),
        generateParams<element::Type_t::i64, element::Type_t::i64>(),
        generateParams<element::Type_t::i64, element::Type_t::u16>(),
        generateParams<element::Type_t::i64, element::Type_t::u32>(),
        generateParams<element::Type_t::i64, element::Type_t::u64>(),
        generateParamsFloat<element::Type_t::i32, element::Type_t::bf16>(),
        generateParamsFloat<element::Type_t::i32, element::Type_t::f16>(),
        generateParamsFloat<element::Type_t::i32, element::Type_t::f32>(),
        generateParamsFloat<element::Type_t::i32, element::Type_t::f64>(),
        generateParamsFloat<element::Type_t::i64, element::Type_t::f32>(),
        generateParamsFloat<element::Type_t::i64, element::Type_t::f64>(),
    };
    std::vector<OneHotParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_With_Hardcoded_Refs,
                         ReferenceOneHotTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceOneHotTest::getTestCaseName);
}  // namespace
