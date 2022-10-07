// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset1.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct EinsumParams {
    std::vector<reference_tests::Tensor> inputs;
    std::string equation;
    reference_tests::Tensor expectedResult;
    std::string testcaseName;
};

struct Builder : ParamsBuilder<EinsumParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, inputs);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, equation);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expectedResult);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, testcaseName);
};

class ReferenceEinsumTest : public testing::TestWithParam<EinsumParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateModel(params);
        for (const auto& input_tensor : params.inputs) {
            inputData.push_back(input_tensor.data);
        }
        refOutData = {params.expectedResult.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<EinsumParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.inputs[0].type;
        result << "_iShape=" << param.inputs[0].shape;
        result << "_equation=" << param.equation;
        result << "_eType=" << param.expectedResult.type;
        result << "_eShape=" << param.expectedResult.shape;
        if (param.testcaseName != "") {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateModel(const EinsumParams& params) {
        OutputVector output_vector;
        ParameterVector param_vector;
        for (const auto& input_tensor : params.inputs) {
            auto param = std::make_shared<opset1::Parameter>(input_tensor.type, input_tensor.shape);
            output_vector.push_back(param);
            param_vector.push_back(param);
        }
        const auto einsum = std::make_shared<opset7::Einsum>(output_vector, params.equation);
        const auto f = std::make_shared<Model>(OutputVector{einsum}, param_vector);
        return f;
    }
};

TEST_P(ReferenceEinsumTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<EinsumParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<EinsumParams> params {
        Builder {}
            .inputs({{ET, {1, 2}, std::vector<T>{1, 2}},
                     {ET, {3, 4}, std::vector<T>{3, 4, 5, 6,
                                                 7, 8, 9, 10,
                                                 11, 12, 13, 14}}})
            .equation("ab,cd->abcd")
            .expectedResult({ET, {1, 2, 3, 4}, std::vector<T>{3,  4,  5,  6,  7,  8,  9,  10,
                                                              11, 12, 13, 14, 6,  8,  10, 12,
                                                              14, 16, 18, 20, 22, 24, 26, 28}})
            .testcaseName("einsum_no_reduction"),
        Builder {}
            .inputs({{ET, {1, 2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}}})
            .equation("ijk->kij")
            .expectedResult({ET, {3, 1, 2}, std::vector<T>{1, 4, 2, 5, 3, 6}})
            .testcaseName("einsum_transpose"),

        Builder {}
            .inputs({{ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}}})
            .equation("ab->a")
            .expectedResult({ET, {2}, std::vector<T>{6, 15}})
            .testcaseName("einsum_reduce"),

        Builder {}
            .inputs({{ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}},
                     {ET, {3, 2}, std::vector<T>{1, 2, 3, 4, 5, 6}}})
            .equation("ab,bc->ac")
            .expectedResult({ET, {2, 2}, std::vector<T>{22, 28, 49, 64}})
            .testcaseName("einsum_matrix_multiplication"),

        Builder {}
            .inputs({{ET, {2, 4}, std::vector<T>{1, 3, 2, 7, 5, 6, 0, 1}},
                     {ET, {4, 3, 1}, std::vector<T>{1, 2, 3, 4, 5, 6, 5, 7, 3, 7, 9, 1}},
                     {ET, {4, 3}, std::vector<T>{4, 3, 1, 6, 4, 2, 2, 5, 3, 1, 9, 4}}})
            .equation("ab,bcd,bc->ca")
            .expectedResult({ET, {3, 2}, std::vector<T>{145, 171, 703, 231, 85, 91}})
            .testcaseName("einsum_multiple_multiplication"),

        Builder {}
            .inputs({{ET, {2, 2, 3}, std::vector<T>{1, 3, 2, 7, 5, 6, 3, 5, 2, 1, 0, 7}}})
            .equation("a...->...")
            .expectedResult({ET, {2, 3}, std::vector<T>{4, 8, 4, 8, 5, 13}})
            .testcaseName("einsum_ellipsis_one_input_reduction"),

        Builder {}
            .inputs({{ET, {2, 2, 3}, std::vector<T>{1, 3, 2, 7, 5, 6, 3, 5, 2, 1, 0, 7}}})
            .equation("a...->...a")
            .expectedResult({ET, {2, 3, 2}, std::vector<T>{1, 3, 3, 5, 2, 2, 7, 1, 5, 0, 6, 7}})
            .testcaseName("einsum_ellipsis_one_input_transpose"),

        Builder {}
            .inputs({{ET, {2, 2, 3}, std::vector<T>{1, 3, 2, 7, 5, 6, 3, 5, 2, 1, 0, 7}},
                     {ET, {1}, std::vector<T>{2}}})
            .equation("ab...,...->ab...")
            .expectedResult({ET, {2, 2, 3}, std::vector<T>{2, 6, 4, 14, 10, 12, 6, 10, 4, 2, 0, 14}})
            .testcaseName("einsum_ellipsis_mul_by_1dscalar"),

        Builder {}
            .inputs({{ET, {1, 1, 4, 3}, std::vector<T>{1, 3, 2, 7, 5, 6, 3, 5, 2, 1, 0, 7}},
                     {ET, {3, 4, 2, 1}, std::vector<T>{3, 1, 6, 2, 3, 10, 9,  8, 2, 9, 3, 2,
                                                       4, 2, 3, 1, 9, 1,  11, 4, 7, 2, 3, 1}}})
            .equation("a...j,j...->a...")
            .expectedResult({ET, {1, 4, 2, 4}, std::vector<T>{27, 85,  37, 66, 30, 58, 50, 8,
                                                              37, 123, 55, 83, 16, 48, 24, 30,
                                                              29, 83,  43, 52, 20, 92, 44, 24,
                                                              24, 96,  48, 30, 13, 67, 31, 15}})
            .testcaseName("einsum_ellipsis_complex_mul"),

        Builder {}
            .inputs({{ET, {1, 3, 3}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9}}})
            .equation("kii->ki")
            .expectedResult({ET, {1, 3}, std::vector<T>{1, 5, 9}})
            .testcaseName("einsum_diagonal"),

        Builder {}
            .inputs({{ET, {2, 3, 3, 2, 4}, std::vector<T>{4, 2, 5, 4, 5, 5, 1, 1, 3, 3, 1, 1, 2, 2, 4, 1, 3, 4,
                                                          4, 5, 1, 3, 1, 3, 1, 4, 3, 5, 4, 4, 5, 4, 4, 5, 4, 2,
                                                          2, 2, 3, 3, 1, 1, 4, 3, 4, 2, 2, 1, 1, 2, 3, 1, 1, 4,
                                                          2, 3, 1, 3, 4, 2, 5, 5, 3, 4, 3, 4, 5, 4, 4, 5, 1, 3,
                                                          4, 4, 5, 3, 1, 3, 2, 5, 3, 2, 5, 4, 4, 2, 4, 4, 1, 4,
                                                          4, 5, 4, 4, 4, 2, 3, 3, 4, 2, 4, 2, 5, 1, 3, 2, 4, 3,
                                                          5, 1, 2, 3, 1, 1, 2, 5, 1, 1, 2, 1, 4, 5, 3, 4, 1, 3,
                                                          3, 1, 3, 2, 4, 5, 1, 1, 5, 4, 5, 2, 2, 3, 3, 1, 2, 4}},
                     {ET, {3, 2, 1}, std::vector<T>{1, 4, 4, 5, 3, 3}}})
            .equation("abbac,bad->ad")
            .expectedResult({ET, {2, 1}, std::vector<T>{123, 129}})
            .testcaseName("einsum_diagonal_with_matmul"),
    };
    return params;
}

std::vector<EinsumParams> generateCombinedParams() {
    const std::vector<std::vector<EinsumParams>> generatedParams {
        generateParams<element::Type_t::i32>(),
        generateParams<element::Type_t::f32>(),
    };
    std::vector<EinsumParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Einsum_With_Hardcoded_Refs, ReferenceEinsumTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceEinsumTest::getTestCaseName);
} // namespace
