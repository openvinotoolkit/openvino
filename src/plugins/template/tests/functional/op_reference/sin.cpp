// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sin.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;

namespace reference_tests {
namespace {

struct SinParams {
    reference_tests::Tensor input;
    reference_tests::Tensor expected;
};

struct Builder : ParamsBuilder<SinParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceSinLayerTest : public testing::TestWithParam<SinParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SinParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::v0::Parameter>(type, shape);
        const auto Sin = std::make_shared<op::v0::Sin>(in);
        return std::make_shared<ov::Model>(NodeVector{Sin}, ParameterVector{in});
    }
};

TEST_P(ReferenceSinLayerTest, SinWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Sin_With_Hardcoded_Refs,
    ReferenceSinLayerTest,
    ::testing::Values(
        Builder{}
            .input({{11},
                    element::f16,
                    std::vector<ov::float16>{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f}})
            .expected({{11},
                       element::f16,
                       std::vector<ov::float16>{0.00000000f,
                                                0.24740396f,
                                                -0.24740396f,
                                                0.47942554f,
                                                -0.47942554f,
                                                0.84147098f,
                                                -0.84147098f,
                                                0.90929743f,
                                                -0.90929743f,
                                                -0.75680250f,
                                                0.75680250f}}),
        Builder{}
            .input({{11},
                    element::f32,
                    std::vector<float>{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f}})
            .expected({{11},
                       element::f32,
                       std::vector<float>{0.00000000f,
                                          0.24740396f,
                                          -0.24740396f,
                                          0.47942554f,
                                          -0.47942554f,
                                          0.84147098f,
                                          -0.84147098f,
                                          0.90929743f,
                                          -0.90929743f,
                                          -0.75680250f,
                                          0.75680250f}}),
        Builder{}
            .input({{7}, element::i32, std::vector<int32_t>{0, 1, -1, 2, -2, 4, -4}})
            .expected({{7}, element::i32, std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0}}),
        Builder{}
            .input({{7}, element::i64, std::vector<int64_t>{0, 1, -1, 2, -2, 4, -4}})
            .expected({{7}, element::i64, std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0}}),
        Builder{}
            .input({{4}, element::u32, std::vector<uint32_t>{0, 1, 2, 4}})
            .expected({{4}, element::u32, std::vector<uint32_t>{0, 0, 0, 0}}),
        Builder{}
            .input({{4}, element::u64, std::vector<uint64_t>{0, 1, 2, 4}})
            .expected({{4}, element::u64, std::vector<uint64_t>{0, 0, 0, 0}})),

    ReferenceSinLayerTest::getTestCaseName);
}  // namespace reference_tests
