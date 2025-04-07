// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/asinh.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;

namespace reference_tests {
namespace {

struct AsinhParams {
    reference_tests::Tensor input;
    reference_tests::Tensor expected;
};

struct Builder : ParamsBuilder<AsinhParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceAsinhLayerTest : public testing::TestWithParam<AsinhParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<AsinhParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::v0::Parameter>(type, shape);
        const auto Asinh = std::make_shared<op::v3::Asinh>(in);
        return std::make_shared<ov::Model>(NodeVector{Asinh}, ParameterVector{in});
    }
};

TEST_P(ReferenceAsinhLayerTest, AsinhWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Asinh_With_Hardcoded_Refs,
    ReferenceAsinhLayerTest,
    ::testing::Values(
        Builder{}
            .input({{11},
                    element::f16,
                    std::vector<ov::float16>{0.f, 1.f, -1.f, 2.f, -2.f, 3.f, -3.f, 4.f, 5.f, 10.f, 100.f}})
            .expected({{11},
                       element::f16,
                       std::vector<ov::float16>{0.00000000f,
                                                0.88137359f,
                                                -0.88137359f,
                                                1.44363548f,
                                                -1.44363548f,
                                                1.81844646f,
                                                -1.81844646f,
                                                2.09471255f,
                                                2.31243834f,
                                                2.99822295f,
                                                5.29834237f}}),
        Builder{}
            .input(
                {{11}, element::f32, std::vector<float>{0.f, 1.f, -1.f, 2.f, -2.f, 3.f, -3.f, 4.f, 5.f, 10.f, 100.f}})
            .expected({{11},
                       element::f32,
                       std::vector<float>{0.00000000f,
                                          0.88137359f,
                                          -0.88137359f,
                                          1.44363548f,
                                          -1.44363548f,
                                          1.81844646f,
                                          -1.81844646f,
                                          2.09471255f,
                                          2.31243834f,
                                          2.99822295f,
                                          5.29834237f}}),
        Builder{}
            .input({{11}, element::i32, std::vector<int32_t>{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}})
            .expected({{11}, element::i32, std::vector<int32_t>{-2, -2, -2, -1, -1, 0, 1, 1, 2, 2, 2}}),
        Builder{}
            .input({{11}, element::i64, std::vector<int64_t>{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}})
            .expected({{11}, element::i64, std::vector<int64_t>{-2, -2, -2, -1, -1, 0, 1, 1, 2, 2, 2}}),
        Builder{}
            .input({{6}, element::u32, std::vector<uint32_t>{0, 1, 2, 3, 4, 5}})
            .expected({{6}, element::u32, std::vector<uint32_t>{0, 1, 1, 2, 2, 2}}),
        Builder{}
            .input({{6}, element::u64, std::vector<uint64_t>{0, 1, 2, 3, 4, 5}})
            .expected({{6}, element::u64, std::vector<uint64_t>{0, 1, 1, 2, 2, 2}})),

    ReferenceAsinhLayerTest::getTestCaseName);
}  // namespace reference_tests
