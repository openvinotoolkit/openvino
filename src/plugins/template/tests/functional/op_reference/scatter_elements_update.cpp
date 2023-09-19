// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_elements_update.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ScatterElementsUpdateParams {
    ScatterElementsUpdateParams(const reference_tests::Tensor& paramData,
                                const reference_tests::Tensor& paramIndices,
                                const reference_tests::Tensor& paramUpdates,
                                const reference_tests::Tensor& paramAxis,
                                const reference_tests::Tensor& paramExpected)
        : input(paramData),
          indices(paramIndices),
          updates(paramUpdates),
          axis(paramAxis),
          expected(paramExpected) {}

    reference_tests::Tensor input;
    reference_tests::Tensor indices;
    reference_tests::Tensor updates;
    reference_tests::Tensor axis;
    reference_tests::Tensor expected;
};

class ReferenceScatterElementsUpdateLayerTest : public testing::TestWithParam<ScatterElementsUpdateParams>,
                                                public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.input.data, params.indices.data, params.updates.data, params.axis.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ScatterElementsUpdateParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "data_sh=" << param.input.shape;
        result << "_data_pr=" << param.input.type;
        result << "_indx_sh=" << param.indices.shape;
        result << "_indx_pr=" << param.indices.type;
        result << "_updt_sh=" << param.updates.shape;
        result << "_updt_pr=" << param.updates.type;
        result << "_axis_sh=" << param.axis.shape;
        result << "_axis_pr=" << param.axis.type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ScatterElementsUpdateParams& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.input.type, params.input.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.indices.type, params.indices.shape);
        const auto C = std::make_shared<op::v0::Parameter>(params.updates.type, params.updates.shape);
        const auto D = std::make_shared<op::v0::Parameter>(params.axis.type, params.axis.shape);
        auto scatterElts = std::make_shared<op::v3::ScatterElementsUpdate>(A, B, C, D);
        return std::make_shared<ov::Model>(NodeVector{scatterElts}, ParameterVector{A, B, C, D});
    }
};

TEST_P(ReferenceScatterElementsUpdateLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_IND>
std::vector<ScatterElementsUpdateParams> generateScatterParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_INT = typename element_type_traits<ET_IND>::value_type;
    std::vector<ScatterElementsUpdateParams> scatterParams{
        // axis = 0
        ScatterElementsUpdateParams(
            reference_tests::Tensor({2, 2}, element::Type(ET), std::vector<T>{1, 2, 3, 4}),          // input
            reference_tests::Tensor({2, 2}, element::Type(ET_IND), std::vector<T_INT>{1, 1, 0, 0}),  // indices
            reference_tests::Tensor({2, 2}, element::Type(ET), std::vector<T>{10, 20, 30, 40}),      // updates
            reference_tests::Tensor({1}, element::Type(ET_IND), std::vector<T_INT>{0}),              // axis
            reference_tests::Tensor({2, 2}, element::Type(ET), std::vector<T>{30, 40, 10, 20})),     // expected
        // axis = 1
        ScatterElementsUpdateParams(
            reference_tests::Tensor({2, 1}, element::Type(ET), std::vector<T>{1, 2}),          // input
            reference_tests::Tensor({2, 1}, element::Type(ET_IND), std::vector<T_INT>{0, 0}),  // indices
            reference_tests::Tensor({2, 1}, element::Type(ET), std::vector<T>{10, 20}),        // updates
            reference_tests::Tensor({1}, element::Type(ET_IND), std::vector<T_INT>{1}),        // axis
            reference_tests::Tensor({2, 1}, element::Type(ET), std::vector<T>{10, 20})),       // expected
    };
    return scatterParams;
}

std::vector<ScatterElementsUpdateParams> generateScatterCombinedParams() {
    const std::vector<std::vector<ScatterElementsUpdateParams>> scatterTypeParams{
        // i16
        generateScatterParams<element::Type_t::i16, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::u64>(),
        // i32
        generateScatterParams<element::Type_t::i32, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::u64>(),
        // i64
        generateScatterParams<element::Type_t::i64, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::u64>(),
        // u32
        generateScatterParams<element::Type_t::u32, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::u64>(),
        // u64
        generateScatterParams<element::Type_t::u64, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::u64>(),
        // f16
        generateScatterParams<element::Type_t::f16, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::u64>(),
        // f32
        generateScatterParams<element::Type_t::f32, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::u64>(),
    };
    std::vector<ScatterElementsUpdateParams> combinedParams;
    for (const auto& params : scatterTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}
INSTANTIATE_TEST_SUITE_P(smoke_ScatterEltsUpdate_With_Hardcoded_Refs,
                         ReferenceScatterElementsUpdateLayerTest,
                         ::testing::ValuesIn(generateScatterCombinedParams()),
                         ReferenceScatterElementsUpdateLayerTest::getTestCaseName);
}  // namespace
