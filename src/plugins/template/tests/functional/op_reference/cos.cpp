// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cos.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;

namespace reference_tests {
namespace {

struct CosParams {
    reference_tests::Tensor input;
    reference_tests::Tensor expected;
};

struct Builder : ParamsBuilder<CosParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceCosLayerTest : public testing::TestWithParam<CosParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<CosParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::v0::Parameter>(type, shape);
        const auto Cos = std::make_shared<op::v0::Cos>(in);
        return std::make_shared<ov::Model>(NodeVector{Cos}, ParameterVector{in});
    }
};

TEST_P(ReferenceCosLayerTest, CosWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Cos_With_Hardcoded_Refs,
    ReferenceCosLayerTest,
    ::testing::Values(
        Builder{}
            .input({{11},
                    element::f16,
                    std::vector<ov::float16>{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f}})
            .expected({{11},
                       element::f16,
                       std::vector<ov::float16>{1.00000000f,
                                                0.96891242f,
                                                0.96891242f,
                                                0.87758256f,
                                                0.87758256f,
                                                0.54030231f,
                                                0.54030231f,
                                                -0.41614684f,
                                                -0.41614684f,
                                                -0.65364362f,
                                                -0.65364362f}}),
        Builder{}
            .input({{11},
                    element::f32,
                    std::vector<float>{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f}})
            .expected({{11},
                       element::f32,
                       std::vector<float>{1.00000000f,
                                          0.96891242f,
                                          0.96891242f,
                                          0.87758256f,
                                          0.87758256f,
                                          0.54030231f,
                                          0.54030231f,
                                          -0.41614684f,
                                          -0.41614684f,
                                          -0.65364362f,
                                          -0.65364362f}}),
        Builder{}
            .input({{5}, element::i32, std::vector<int32_t>{1, 2, 3, 4, 5}})
            .expected({{5}, element::i32, std::vector<int32_t>{1, 0, -1, -1, 0}}),
        Builder{}
            .input({{5}, element::i64, std::vector<int64_t>{1, 2, 3, 4, 5}})
            .expected({{5}, element::i64, std::vector<int64_t>{1, 0, -1, -1, 0}}),
        Builder{}
            .input({{3}, element::u32, std::vector<uint32_t>{1, 2, 5}})
            .expected({{3}, element::u32, std::vector<uint32_t>{1, 0, 0}}),
        Builder{}
            .input({{3}, element::u64, std::vector<uint64_t>{1, 2, 5}})
            .expected({{3}, element::u64, std::vector<uint64_t>{1, 0, 0}})),

    ReferenceCosLayerTest::getTestCaseName);
}  // namespace reference_tests
