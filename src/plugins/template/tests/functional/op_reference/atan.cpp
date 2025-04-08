// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/atan.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;

namespace reference_tests {
namespace {

struct AtanParams {
    reference_tests::Tensor input;
    reference_tests::Tensor expected;
};

struct Builder : ParamsBuilder<AtanParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceAtanLayerTest : public testing::TestWithParam<AtanParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<AtanParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::v0::Parameter>(type, shape);
        const auto atan = std::make_shared<op::v0::Atan>(in);
        return std::make_shared<ov::Model>(NodeVector{atan}, ParameterVector{in});
    }
};

TEST_P(ReferenceAtanLayerTest, AtanWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Atan_With_Hardcoded_Refs,
    ReferenceAtanLayerTest,
    ::testing::Values(
        Builder{}
            .input({{11},
                    element::f16,
                    std::vector<ov::float16>{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f}})
            .expected({{11},
                       element::f16,
                       std::vector<ov::float16>{-1.32581766f,
                                                -1.10714872f,
                                                -0.78539816f,
                                                -0.46364761f,
                                                -0.24497866f,
                                                0.00000000f,
                                                0.24497866f,
                                                0.46364761f,
                                                0.78539816f,
                                                1.10714872f,
                                                1.32581766f}}),
        Builder{}
            .input({{11},
                    element::f32,
                    std::vector<float>{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f}})
            .expected({{11},
                       element::f32,
                       std::vector<float>{-1.32581766f,
                                          -1.10714872f,
                                          -0.78539816f,
                                          -0.46364761f,
                                          -0.24497866f,
                                          0.00000000f,
                                          0.24497866f,
                                          0.46364761f,
                                          0.78539816f,
                                          1.10714872f,
                                          1.32581766f}}),
        Builder{}
            .input({{5}, element::i32, std::vector<int32_t>{-2, -1, 0, 1, 2}})
            .expected({{5}, element::i32, std::vector<int32_t>{-1, -1, 0, 1, 1}}),
        Builder{}
            .input({{5}, element::i64, std::vector<int64_t>{-2, -1, 0, 1, 2}})
            .expected({{5}, element::i64, std::vector<int64_t>{-1, -1, 0, 1, 1}}),
        Builder{}
            .input({{5}, element::u32, std::vector<uint32_t>{0, 1, 2, 3, 4}})
            .expected({{5}, element::u32, std::vector<uint32_t>{0, 1, 1, 1, 1}}),
        Builder{}
            .input({{5}, element::u64, std::vector<uint64_t>{0, 1, 2, 3, 4}})
            .expected({{5}, element::u64, std::vector<uint64_t>{0, 1, 1, 1, 1}})),
    ReferenceAtanLayerTest::getTestCaseName);
}  // namespace reference_tests
