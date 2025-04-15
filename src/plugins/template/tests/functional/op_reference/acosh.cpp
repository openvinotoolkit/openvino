// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/acosh.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;

namespace reference_tests {
namespace {

struct AcoshParams {
    reference_tests::Tensor input;
    reference_tests::Tensor expected;
};

struct Builder : ParamsBuilder<AcoshParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceAcoshLayerTest : public testing::TestWithParam<AcoshParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<AcoshParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::v0::Parameter>(type, shape);
        const auto acosh = std::make_shared<op::v3::Acosh>(in);
        return std::make_shared<ov::Model>(NodeVector{acosh}, ParameterVector{in});
    }
};

TEST_P(ReferenceAcoshLayerTest, AcoshWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Acosh_With_Hardcoded_Refs,
    ReferenceAcoshLayerTest,
    ::testing::Values(
        Builder{}
            .input({{8}, element::f16, std::vector<ov::float16>{1.f, 2.f, 3.f, 4.f, 5.f, 10.f, 100.f, 1000.f}})
            .expected(
                {{8}, element::f16, std::vector<ov::float16>{0., 1.317, 1.763, 2.063, 2.292, 2.993, 5.298, 7.6012}}),
        Builder{}
            .input({{8}, element::f32, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 10.f, 100.f, 1000.f}})
            .expected({{8}, element::f32, std::vector<float>{0., 1.317, 1.763, 2.063, 2.292, 2.993, 5.298, 7.6012}}),
        Builder{}
            .input({{8}, element::i32, std::vector<int32_t>{1, 2, 3, 4, 5, 10, 100, 1000}})
            .expected({{8}, element::i32, std::vector<int32_t>{0, 1, 2, 2, 2, 3, 5, 8}}),
        Builder{}
            .input({{8}, element::i64, std::vector<int64_t>{1, 2, 3, 4, 5, 10, 100, 1000}})
            .expected({{8}, element::i64, std::vector<int64_t>{0, 1, 2, 2, 2, 3, 5, 8}}),
        Builder{}
            .input({{8}, element::u32, std::vector<uint32_t>{1, 2, 3, 4, 5, 10, 100, 1000}})
            .expected({{8}, element::u32, std::vector<uint32_t>{0, 1, 2, 2, 2, 3, 5, 8}}),
        Builder{}
            .input({{8}, element::u64, std::vector<uint64_t>{1, 2, 3, 4, 5, 10, 100, 1000}})
            .expected({{8}, element::u64, std::vector<uint64_t>{0, 1, 2, 2, 2, 3, 5, 8}})),
    ReferenceAcoshLayerTest::getTestCaseName);
}  // namespace reference_tests
