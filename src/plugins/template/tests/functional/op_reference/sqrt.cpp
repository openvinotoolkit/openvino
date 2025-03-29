// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sqrt.hpp"

#include <gtest/gtest.h>

#include <string>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct SqrtParams {
    reference_tests::Tensor input;
    reference_tests::Tensor expected;
    std::string test_case_name;
};

struct Builder : ParamsBuilder<SqrtParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, test_case_name);
};

class ReferenceSqrt : public testing::TestWithParam<SqrtParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.input);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SqrtParams>& obj) {
        return obj.param.test_case_name;
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& input) {
        const auto in = std::make_shared<op::v0::Parameter>(input.type, input.shape);
        const auto sqrt = std::make_shared<op::v0::Sqrt>(in);
        return std::make_shared<Model>(NodeVector{sqrt}, ParameterVector{in});
    }
};

}  // namespace

TEST_P(ReferenceSqrt, LayerTest) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    ReferenceSqrt,
    ::testing::Values(
        Builder{}
            .input({Shape{2, 3}, element::f32, std::vector<float>{16, 4, 81, 100, 10000, 0}})
            .expected({Shape{2, 3}, element::f32, std::vector<float>{4, 2, 9, 10, 100, 0}})
            .test_case_name("basic"),
        Builder{}
            .input({Shape{4}, element::f32, std::vector<float>{-1, 4, -81, 100}})
            .expected({Shape{4}, element::f32, std::vector<float>{NAN, 2, NAN, 10}})
            .test_case_name("negative_inputs"),
        Builder{}
            .input({Shape{2, 7},
                    element::i32,
                    std::vector<int>{4, 7, 9, 10, 80, 55, 6, 1, 23, 233, 256, 474, 1024, 110889}})
            .expected({Shape{2, 7}, element::i32, std::vector<int>{2, 3, 3, 3, 9, 7, 2, 1, 5, 15, 16, 22, 32, 333}})
            .test_case_name("integral_inputs"),
        Builder{}
            .input({Shape{2, 7},
                    element::f32,
                    std::vector<float>{4, 7, 9, 10, 80, 55, 6.25, 0.9, 23.33, 233, 256, 473.7891, 1024, 111108.88}})
            .expected({Shape{2, 7},
                       element::f32,
                       std::vector<float>{2.,
                                          2.6457512,
                                          3.,
                                          3.1622777,
                                          8.944272,
                                          7.4161983,
                                          2.5,
                                          0.94868326,
                                          4.830114,
                                          15.264338,
                                          16.,
                                          21.766697,
                                          32.,
                                          333.33}})
            .test_case_name("floating_inputs")),
    ReferenceSqrt::getTestCaseName);
