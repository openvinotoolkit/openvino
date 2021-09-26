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

struct CeilingParams {
    Tensor input;
    Tensor expected;
};

struct Builder : ParamsBuilder<CeilingParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceCeilingLayerTest : public testing::TestWithParam<CeilingParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<CeilingParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::Parameter>(type, shape);
        const auto ceiling = std::make_shared<op::Ceiling>(in);
        return std::make_shared<Function>(NodeVector {ceiling}, ParameterVector {in});
    }
};

TEST_P(ReferenceCeilingLayerTest, CeilingWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Ceiling_With_Hardcoded_Refs, ReferenceCeilingLayerTest,
    ::testing::Values(Builder {}
                          .input({{4}, element::f16, std::vector<ngraph::float16>{-2.5f, -2.0f, 0.3f, 4.8f}})
                          .expected({{4}, element::f16, std::vector<ngraph::float16> {-2.0f, -2.0f, 1.0f, 5.0f}}),
                      Builder {}
                          .input({{4}, element::f32, std::vector<float> {-2.5f, -2.0f, 0.3f, 4.8f}})
                          .expected({{4}, element::f32, std::vector<float> {-2.0f, -2.0f, 1.0f, 5.0f}}),
                      Builder {}
                          .input({{4}, element::i32, std::vector<int32_t> {-2, -136314888, 0x40000010, 0x40000001}})
                          .expected({{4}, element::i32, std::vector<int32_t> {-2, -136314888, 0x40000010, 0x40000001}}),
                      Builder {}
                          .input({{3}, element::i64, std::vector<int64_t> {0, 1, 0x4000000000000001}})
                          .expected({{3}, element::i64, std::vector<int64_t> {0, 1, 0x4000000000000001}}),
                      Builder {}
                          .input({{4}, element::u32, std::vector<uint32_t> {2, 136314888, 0x40000010, 0x40000001}})
                          .expected({{4}, element::u32, std::vector<uint32_t> {2, 136314888, 0x40000010, 0x40000001}}),
                      Builder {}
                          .input({{3}, element::u64, std::vector<uint64_t> {0, 1, 0x4000000000000001}})
                          .expected({{3}, element::u64, std::vector<uint64_t>{0, 1, 0x4000000000000001}})),
    ReferenceCeilingLayerTest::getTestCaseName);
}  // namespace reference_tests
