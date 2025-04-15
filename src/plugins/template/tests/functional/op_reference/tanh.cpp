// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tanh.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;

namespace reference_tests {
namespace {

struct TanhParams {
    reference_tests::Tensor input;
    reference_tests::Tensor expected;
};

struct Builder : ParamsBuilder<TanhParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceTanhLayerTest : public testing::TestWithParam<TanhParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<TanhParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::v0::Parameter>(type, shape);
        const auto Tanh = std::make_shared<op::v0::Tanh>(in);
        return std::make_shared<ov::Model>(NodeVector{Tanh}, ParameterVector{in});
    }
};

TEST_P(ReferenceTanhLayerTest, TanhWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Tanh_With_Hardcoded_Refs,
    ReferenceTanhLayerTest,
    ::testing::Values(
        Builder{}
            .input({{11},
                    element::f16,
                    std::vector<ov::float16>{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f}})
            .expected({{11},
                       element::f16,
                       std::vector<ov::float16>{-0.99932930f,
                                                -0.96402758f,
                                                -0.76159416f,
                                                -0.46211716f,
                                                -0.24491866f,
                                                0.00000000f,
                                                0.24491866f,
                                                0.46211716f,
                                                0.76159416f,
                                                0.96402758f,
                                                0.99932930f}}),
        Builder{}
            .input({{11},
                    element::f32,
                    std::vector<float>{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f}})
            .expected({{11},
                       element::f32,
                       std::vector<float>{-0.99932930f,
                                          -0.96402758f,
                                          -0.76159416f,
                                          -0.46211716f,
                                          -0.24491866f,
                                          0.00000000f,
                                          0.24491866f,
                                          0.46211716f,
                                          0.76159416f,
                                          0.96402758f,
                                          0.99932930f}}),
        Builder{}
            .input({{7}, element::i32, std::vector<int32_t>{-4, -2, -1, 0, 1, 2, 4}})
            .expected({{7}, element::i32, std::vector<int32_t>{-1, -1, -1, 0, 1, 1, 1}}),
        Builder{}
            .input({{7}, element::i64, std::vector<int64_t>{-4, -2, -1, 0, 1, 2, 4}})
            .expected({{7}, element::i64, std::vector<int64_t>{-1, -1, -1, 0, 1, 1, 1}}),
        Builder{}
            .input({{4}, element::u32, std::vector<uint32_t>{0, 1, 2, 4}})
            .expected({{4}, element::u32, std::vector<uint32_t>{0, 1, 1, 1}}),
        Builder{}
            .input({{4}, element::u64, std::vector<uint64_t>{0, 1, 2, 4}})
            .expected({{4}, element::u64, std::vector<uint64_t>{0, 1, 1, 1}})),

    ReferenceTanhLayerTest::getTestCaseName);
}  // namespace reference_tests
