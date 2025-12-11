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
struct OneHotParams_v1 {
    OneHotParams_v1(const reference_tests::Tensor& dataTensor,
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

    std::string ToStr() const {
        std::ostringstream result;
        result << "dType=" << dataTensor.type;
        result << "_dShape=" << dataTensor.shape;
        result << "_axis=" << axis;
        result << "_deType=" << depthTensor.type;
        result << "_deShape=" << depthTensor.shape;
        result << "_onType=" << onValueTensor.type;
        result << "_onShape=" << onValueTensor.shape;
        result << "_offType=" << offValueTensor.type;
        result << "_offShape=" << offValueTensor.shape;
        result << "_eType=" << expectedTensor.type;
        if (testcaseName != "") {
            result << "_eShape=" << expectedTensor.shape;
            result << "_=" << testcaseName;
        } else {
            result << "_eShape=" << expectedTensor.shape;
        }
        return result.str();
    }

    reference_tests::Tensor dataTensor;
    int32_t axis;
    reference_tests::Tensor depthTensor;
    reference_tests::Tensor onValueTensor;
    reference_tests::Tensor offValueTensor;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

struct OneHotParams_v16 : public OneHotParams_v1 {
    OneHotParams_v16(const reference_tests::Tensor& dataTensor,
                     const int32_t axis,
                     const reference_tests::Tensor& depthTensor,
                     const reference_tests::Tensor& onValueTensor,
                     const reference_tests::Tensor& offValueTensor,
                     const reference_tests::Tensor& expectedTensor,
                     ov::op::v16::OneHot::NegativeIndicesMode mode,
                     const std::string& testcaseName = "")
        : OneHotParams_v1(dataTensor, axis, depthTensor, onValueTensor, offValueTensor, expectedTensor, testcaseName),
          mode(mode) {}

    std::string ToStr() const {
        std::string result_v1 = OneHotParams_v1::ToStr();
        if (mode == op::v16::OneHot::NegativeIndicesMode::NORMALIZE) {
            result_v1 += "_mode=NORMALIZE";
        } else {
            result_v1 += "_mode=IGNORE_NEGATIVE";
        }
        return result_v1;
    }

    static OneHotParams_v16 From_v1(const OneHotParams_v1& params) {
        return OneHotParams_v16(params.dataTensor,
                                params.axis,
                                params.depthTensor,
                                params.onValueTensor,
                                params.offValueTensor,
                                params.expectedTensor,
                                ov::op::v16::OneHot::NegativeIndicesMode::IGNORE_NEGATIVE,
                                params.testcaseName);
    }

    op::v16::OneHot::NegativeIndicesMode mode;
};

template <class TParams>
class ReferenceOneHotTest_Base : public testing::TestWithParam<TParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = this->GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<TParams>& obj) {
        auto param = obj.param;
        return param.ToStr();
    }

    virtual std::shared_ptr<Model> CreateFunction(const TParams& params) = 0;
};

class ReferenceOneHotTest_v1 : public ReferenceOneHotTest_Base<OneHotParams_v1> {
public:
    std::shared_ptr<Model> CreateFunction(const OneHotParams_v1& params) override {
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

class ReferenceOneHotTest_v16 : public ReferenceOneHotTest_Base<OneHotParams_v16> {
public:
    std::shared_ptr<Model> CreateFunction(const OneHotParams_v16& params) override {
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
        const auto oneHot = std::make_shared<op::v16::OneHot>(data, depth, onValue, offValue, params.axis, params.mode);
        function = std::make_shared<ov::Model>(oneHot, ParameterVector{data});
        return function;
    }
};

TEST_P(ReferenceOneHotTest_v1, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceOneHotTest_v16, CompareWithRefs) {
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
std::vector<OneHotParams_v1> generateParams_v1() {
    using T1 = typename element_type_traits<ET1>::value_type;
    using T2 = typename element_type_traits<ET2>::value_type;
    std::vector<OneHotParams_v1> params{
        OneHotParams_v1(reference_tests::Tensor(ET1, {}, std::vector<T1>{2}),
                        0,
                        reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                        reference_tests::Tensor(ET2, {3}, std::vector<T2>{0, 0, 1}),
                        "one_hot_scalar_2_in_3"),
        OneHotParams_v1(reference_tests::Tensor(ET1, {}, std::vector<T1>{1}),
                        0,
                        reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                        reference_tests::Tensor(ET2, {3}, std::vector<T2>{0, 1, 0}),
                        "one_hot_scalar_1_in_3"),
        OneHotParams_v1(reference_tests::Tensor(ET1, {}, std::vector<T1>{0}),
                        0,
                        reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                        reference_tests::Tensor(ET2, {3}, std::vector<T2>{1, 0, 0}),
                        "one_hot_scalar_0_in_3"),
        OneHotParams_v1(reference_tests::Tensor(ET1, {8}, std::vector<T1>{2, 1, 0, 0, 2, 2, 1, 0}),
                        0,
                        reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                        reference_tests::Tensor(ET2, {3, 8}, std::vector<T2>{0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
                                                                             0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0}),
                        "one_hot_vector_0"),
        OneHotParams_v1(reference_tests::Tensor(ET1, {8}, std::vector<T1>{2, 1, 0, 0, 2, 2, 1, 0}),
                        1,
                        reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                        reference_tests::Tensor(ET2, {8, 3}, std::vector<T2>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                                                                             0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
                        "one_hot_vector_1"),
        OneHotParams_v1(reference_tests::Tensor(ET1, {8}, std::vector<T1>{2, 1, 0, 0, 3, 2, 1, 0}),
                        1,
                        reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
                        reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
                        reference_tests::Tensor(ET2, {8, 3}, std::vector<T2>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                                                                             0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
                        "one_hot_vector_1_barely_oob"),
        OneHotParams_v1(
            reference_tests::Tensor(ET1, {3, 3}, std::vector<T1>{0, 1, 1, 2, 1, 0, 0, 2, 1}),
            0,
            reference_tests::Tensor(ET1, {}, std::vector<T1>{3}),
            reference_tests::Tensor(ET2, {}, std::vector<T2>{1}),
            reference_tests::Tensor(ET2, {}, std::vector<T2>{0}),
            reference_tests::Tensor(ET2, {3, 3, 3}, std::vector<T2>{1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,
                                                                    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0}),
            "one_hot_matrix_0"),
        OneHotParams_v1(reference_tests::Tensor(ET1, {6}, std::vector<T1>{0, 11, 101, 1001, 10001, 19999}),
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
std::vector<OneHotParams_v1> generateParams_Float_v1() {
    using T1 = typename element_type_traits<ET1>::value_type;
    using T2 = typename element_type_traits<ET2>::value_type;
    std::vector<OneHotParams_v1> params{
        OneHotParams_v1(
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

std::vector<OneHotParams_v1> generateCombinedParams_v1() {
    const std::vector<std::vector<OneHotParams_v1>> generatedParams{
        generateParams_v1<element::Type_t::i32, element::Type_t::i16>(),
        generateParams_v1<element::Type_t::i32, element::Type_t::i32>(),
        generateParams_v1<element::Type_t::i32, element::Type_t::i64>(),
        generateParams_v1<element::Type_t::i32, element::Type_t::u16>(),
        generateParams_v1<element::Type_t::i32, element::Type_t::u32>(),
        generateParams_v1<element::Type_t::i32, element::Type_t::u64>(),
        generateParams_v1<element::Type_t::i64, element::Type_t::i16>(),
        generateParams_v1<element::Type_t::i64, element::Type_t::i32>(),
        generateParams_v1<element::Type_t::i64, element::Type_t::i64>(),
        generateParams_v1<element::Type_t::i64, element::Type_t::u16>(),
        generateParams_v1<element::Type_t::i64, element::Type_t::u32>(),
        generateParams_v1<element::Type_t::i64, element::Type_t::u64>(),
        generateParams_Float_v1<element::Type_t::i32, element::Type_t::bf16>(),
        generateParams_Float_v1<element::Type_t::i32, element::Type_t::f16>(),
        generateParams_Float_v1<element::Type_t::i32, element::Type_t::f32>(),
        generateParams_Float_v1<element::Type_t::i32, element::Type_t::f64>(),
        generateParams_Float_v1<element::Type_t::i64, element::Type_t::f32>(),
        generateParams_Float_v1<element::Type_t::i64, element::Type_t::f64>(),
    };
    std::vector<OneHotParams_v1> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

std::vector<OneHotParams_v16> generateCombinedParams_v16() {
    const auto params_v1 = generateCombinedParams_v1();
    std::vector<OneHotParams_v16> params_v16;
    params_v16.reserve(params_v1.size());
    for (const auto& p : params_v1) {
        params_v16.push_back(OneHotParams_v16::From_v1(p));
    }

    params_v16.push_back(OneHotParams_v16(
        reference_tests::Tensor(element::Type_t::i32, {8}, std::vector<int32_t>{-1, -2, 0, 0, 2, -1, 1, 0}),
        0,
        reference_tests::Tensor(element::Type_t::i32, {}, std::vector<int32_t>{3}),
        reference_tests::Tensor(element::Type_t::f32, {}, std::vector<float>{1}),
        reference_tests::Tensor(element::Type_t::f32, {}, std::vector<float>{0}),
        reference_tests::Tensor(element::Type_t::f32, {3, 8}, std::vector<float>{0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
                                                                                 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0}),
        op::v16::OneHot::NegativeIndicesMode::NORMALIZE,
        "one_hot_normalize_mode"));
    return params_v16;
}

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_With_Hardcoded_Refs,
                         ReferenceOneHotTest_v1,
                         testing::ValuesIn(generateCombinedParams_v1()),
                         ReferenceOneHotTest_v1::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_With_Hardcoded_Refs,
                         ReferenceOneHotTest_v16,
                         testing::ValuesIn(generateCombinedParams_v16()),
                         ReferenceOneHotTest_v16::getTestCaseName);
}  // namespace
