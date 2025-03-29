// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/asin.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;

namespace reference_tests {
namespace {

struct AsinParams {
    reference_tests::Tensor input;
    reference_tests::Tensor expected;
};

struct Builder : ParamsBuilder<AsinParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceAsinLayerTest : public testing::TestWithParam<AsinParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<AsinParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::v0::Parameter>(type, shape);
        const auto Asin = std::make_shared<op::v0::Asin>(in);
        return std::make_shared<ov::Model>(NodeVector{Asin}, ParameterVector{in});
    }
};

TEST_P(ReferenceAsinLayerTest, AsinWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Asin_With_Hardcoded_Refs,
    ReferenceAsinLayerTest,
    ::testing::Values(
        Builder{}
            .input(
                {{11},
                 element::f16,
                 std::vector<ov::float16>{-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f}})
            .expected({{11},
                       element::f16,
                       std::vector<ov::float16>{-1.57079633f,
                                                -0.84806208f,
                                                -0.52359878f,
                                                -0.25268026f,
                                                -0.12532783f,
                                                0.00000000f,
                                                0.12532783f,
                                                0.25268026f,
                                                0.52359878f,
                                                0.84806208f,
                                                1.57079633f}}),
        Builder{}
            .input({{11},
                    element::f32,
                    std::vector<float>{-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f}})
            .expected({{11},
                       element::f32,
                       std::vector<float>{-1.57079633f,
                                          -0.84806208f,
                                          -0.52359878f,
                                          -0.25268026f,
                                          -0.12532783f,
                                          0.00000000f,
                                          0.12532783f,
                                          0.25268026f,
                                          0.52359878f,
                                          0.84806208f,
                                          1.57079633f}}),
        Builder{}
            .input({{3}, element::i32, std::vector<int32_t>{-1, 0, 1}})
            .expected({{3}, element::i32, std::vector<int32_t>{-1, 0, 1}}),
        Builder{}
            .input({{3}, element::i64, std::vector<int64_t>{-1, 0, 1}})
            .expected({{3}, element::i64, std::vector<int64_t>{-1, 0, 1}}),
        Builder{}
            .input({{2}, element::u32, std::vector<uint32_t>{0, 1}})
            .expected({{2}, element::u32, std::vector<uint32_t>{0, 1}}),
        Builder{}
            .input({{2}, element::u64, std::vector<uint64_t>{0, 1}})
            .expected({{2}, element::u64, std::vector<uint64_t>{0, 1}})),

    ReferenceAsinLayerTest::getTestCaseName);
}  // namespace reference_tests
