// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sinh.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;

namespace reference_tests {
namespace {

struct SinhParams {
    reference_tests::Tensor input;
    reference_tests::Tensor expected;
};

struct Builder : ParamsBuilder<SinhParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceSinhLayerTest : public testing::TestWithParam<SinhParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SinhParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::v0::Parameter>(type, shape);
        const auto Sinh = std::make_shared<op::v0::Sinh>(in);
        return std::make_shared<ov::Model>(NodeVector{Sinh}, ParameterVector{in});
    }
};

TEST_P(ReferenceSinhLayerTest, SinhWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Sinh_With_Hardcoded_Refs,
    ReferenceSinhLayerTest,
    ::testing::Values(
        Builder{}
            .input({{11},
                    element::f16,
                    std::vector<ov::float16>{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f}})
            .expected({{11},
                       element::f16,
                       std::vector<ov::float16>{-27.28991720f,
                                                -3.62686041f,
                                                -1.17520120f,
                                                -0.52109531f,
                                                -0.25261232f,
                                                0.00000000f,
                                                0.25261232f,
                                                0.52109531f,
                                                1.17520120f,
                                                3.62686041f,
                                                27.28991720f}}),
        Builder{}
            .input({{11},
                    element::f32,
                    std::vector<float>{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f}})
            .expected({{11},
                       element::f32,
                       std::vector<float>{-27.28991720f,
                                          -3.62686041f,
                                          -1.17520120f,
                                          -0.52109531f,
                                          -0.25261232f,
                                          0.00000000f,
                                          0.25261232f,
                                          0.52109531f,
                                          1.17520120f,
                                          3.62686041f,
                                          27.28991720f}}),
        Builder{}
            .input({{7}, element::i32, std::vector<int32_t>{-4, -2, -1, 0, 1, 2, 4}})
            .expected({{7}, element::i32, std::vector<int32_t>{-27, -4, -1, 0, 1, 4, 27}}),
        Builder{}
            .input({{7}, element::i64, std::vector<int64_t>{-4, -2, -1, 0, 1, 2, 4}})
            .expected({{7}, element::i64, std::vector<int64_t>{-27, -4, -1, 0, 1, 4, 27}}),
        Builder{}
            .input({{4}, element::u32, std::vector<uint32_t>{0, 1, 2, 4}})
            .expected({{4}, element::u32, std::vector<uint32_t>{0, 1, 4, 27}}),
        Builder{}
            .input({{4}, element::u64, std::vector<uint64_t>{0, 1, 2, 4}})
            .expected({{4}, element::u64, std::vector<uint64_t>{0, 1, 4, 27}})),

    ReferenceSinhLayerTest::getTestCaseName);
}  // namespace reference_tests
