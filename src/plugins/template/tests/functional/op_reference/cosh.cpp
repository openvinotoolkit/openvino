// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cosh.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;

namespace reference_tests {
namespace {

struct CoshParams {
    reference_tests::Tensor input;
    reference_tests::Tensor expected;
};

struct Builder : ParamsBuilder<CoshParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceCoshLayerTest : public testing::TestWithParam<CoshParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<CoshParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::v0::Parameter>(type, shape);
        const auto Cosh = std::make_shared<op::v0::Cosh>(in);
        return std::make_shared<ov::Model>(NodeVector{Cosh}, ParameterVector{in});
    }
};

TEST_P(ReferenceCoshLayerTest, CoshWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Cosh_With_Hardcoded_Refs,
    ReferenceCoshLayerTest,
    ::testing::Values(
        Builder{}
            .input({{11},
                    element::f16,
                    std::vector<ov::float16>{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f}})
            .expected({{11},
                       element::f16,
                       std::vector<ov::float16>{27.30823284f,
                                                3.76219569f,
                                                1.54308063f,
                                                1.12762597f,
                                                1.03141310f,
                                                1.00000000f,
                                                1.03141310f,
                                                1.12762597f,
                                                1.54308063f,
                                                3.76219569f,
                                                27.30823284f}}),
        Builder{}
            .input({{11},
                    element::f32,
                    std::vector<float>{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f}})
            .expected({{11},
                       element::f32,
                       std::vector<float>{27.30823284f,
                                          3.76219569f,
                                          1.54308063f,
                                          1.12762597f,
                                          1.03141310f,
                                          1.00000000f,
                                          1.03141310f,
                                          1.12762597f,
                                          1.54308063f,
                                          3.76219569f,
                                          27.30823284f}}),
        Builder{}
            .input({{7}, element::i32, std::vector<int32_t>{-4, -2, -1, 0, 1, 2, 4}})
            .expected({{7}, element::i32, std::vector<int32_t>{27, 4, 2, 1, 2, 4, 27}}),
        Builder{}
            .input({{7}, element::i64, std::vector<int64_t>{-4, -2, -1, 0, 1, 2, 4}})
            .expected({{7}, element::i64, std::vector<int64_t>{27, 4, 2, 1, 2, 4, 27}}),
        Builder{}
            .input({{4}, element::u32, std::vector<uint32_t>{0, 1, 2, 4}})
            .expected({{4}, element::u32, std::vector<uint32_t>{1, 2, 4, 27}}),
        Builder{}
            .input({{4}, element::u64, std::vector<uint64_t>{0, 1, 2, 4}})
            .expected({{4}, element::u64, std::vector<uint64_t>{1, 2, 4, 27}})),

    ReferenceCoshLayerTest::getTestCaseName);
}  // namespace reference_tests
