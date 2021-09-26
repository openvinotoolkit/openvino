// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vector>

#include "base_reference_test.hpp"

using namespace ngraph;

namespace reference_tests {
namespace {

struct AbsParams {
    Tensor input;
    Tensor expected;
};

struct Builder : ParamsBuilder<AbsParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceAbsLayerTest : public testing::TestWithParam<AbsParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<AbsParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::Parameter>(type, shape);
        const auto abs = std::make_shared<op::Abs>(in);
        return std::make_shared<Function>(NodeVector {abs}, ParameterVector {in});
    }
};

TEST_P(ReferenceAbsLayerTest, AbsWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Abs_With_Hardcoded_Refs, ReferenceAbsLayerTest,
    ::testing::Values(Builder {}
                          .input({{4}, element::f16, std::vector<ngraph::float16> {1.f, -2.f, 0.f, -4.75f}})
                          .expected({{4}, element::f16, std::vector<ngraph::float16> {1.f, 2.f, 0.f, 4.75f}}),
                      Builder {}
                          .input({{4}, element::f32, std::vector<float> {1.f, -2.f, 0.f, -4.75f}})
                          .expected({{4}, element::f32, std::vector<float>{1.f, 2.f, 0.f, 4.75f}}),
                      Builder {}
                          .input({{4}, element::i32, std::vector<int32_t> {1, -2, 0, -4}})
                          .expected({{4}, element::i32, std::vector<int32_t>{1, 2, 0, 4}}),
                      Builder {}
                          .input({{4}, element::i64, std::vector<int64_t>{1, -2, 0, -4}})
                          .expected({{4}, element::i64, std::vector<int64_t>{1, 2, 0, 4}}),
                      Builder {}
                          .input({{4}, element::u32, std::vector<uint32_t>{1, 2, 0, 4}})
                          .expected({{4}, element::u32, std::vector<uint32_t>{1, 2, 0, 4}}),
                      Builder {}
                          .input({{4}, element::u64, std::vector<uint64_t>{1, 2, 0, 4}})
                          .expected({{4}, element::u64, std::vector<uint64_t>{1, 2, 0, 4}})),
    ReferenceAbsLayerTest::getTestCaseName);
}  // namespace reference_tests
