// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// TODO: add tests for negative axis

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vector>

#include "base_reference_test.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace reference_tests;

namespace reference_tests {

namespace {

// ------------------------------ V3 ------------------------------

struct ScatterUpdate6Params {
    Tensor data;
    Tensor indices;
    Tensor updates;
    Tensor axis;
    Tensor expected;
};

struct Builder : ParamsBuilder<ScatterUpdate6Params> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, data);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, indices);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, updates);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, axis);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceScatterUpdate6LayerTest : public testing::TestWithParam<ScatterUpdate6Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.data.shape, params.indices.shape, params.updates.shape, params.axis.shape,
                                  params.data.type, params.indices.type, params.axis.type);
        inputData = {params.data.data, params.indices.data, params.updates.data, params.axis.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ScatterUpdate6Params>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "D_shape=" << param.data.shape << "_";
        result << "I_shape=" << param.indices.shape << "_";
        result << "U_shape=" << param.updates.shape << "_";
        result << "A_shape=" << param.axis.shape << "_";
        result << "dType=" << param.data.type << "_";
        result << "iType=" << param.indices.type << "_";
        result << "uType=" << param.updates.type << "_";
        result << "aType=" << param.axis.type << "_";
        result << "oType=" << param.expected.type;
        return result.str();
    }

private:
    static std::shared_ptr<ngraph::Function> CreateFunction(const ngraph::PartialShape& data_shape, const ngraph::PartialShape& indices_shape,
                                                            const ngraph::PartialShape& updates_shape, const ngraph::PartialShape& axis_shape,
                                                            const ngraph::element::Type& numeric_type, const ngraph::element::Type& indices_type,
                                                            const ngraph::element::Type& axis_type) {
        const auto data = std::make_shared<ngraph::op::Parameter>(numeric_type, data_shape);
        const auto indices = std::make_shared<ngraph::op::Parameter>(indices_type, indices_shape);
        const auto updates = std::make_shared<ngraph::op::Parameter>(numeric_type, updates_shape);
        const auto axis = std::make_shared<ngraph::op::Parameter>(axis_type, axis_shape);
        const auto scatter_update = std::make_shared<op::v3::ScatterUpdate>(data, indices, updates, axis);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector {scatter_update}, ngraph::ParameterVector {data, indices, updates, axis});
    }
};

TEST_P(ReferenceScatterUpdate6LayerTest, ScatterUpdateWithHardcodedRefs) {
    Exec();
}

template <element::Type_t NUM_ET, element::Type_t INT_ET>
std::vector<ScatterUpdate6Params> generateScatterUpdate6Params(const element::Type& numeric_type, const element::Type& integer_type) {
    using N = typename element_type_traits<NUM_ET>::value_type;
    using I = typename element_type_traits<INT_ET>::value_type;
    std::vector<ScatterUpdate6Params> ScatterUpdateParams {
        Builder {}
            .data({{3, 2, 2, 3}, numeric_type, std::vector<N> {// ---------------------
                                                               // 1
                                                               0, 0, 0,
                                                               0, 0, 0,
                                                               // 2
                                                               0, 0, 0,
                                                               0, 0, 0,
                                                               // ---------------------
                                                               // 3
                                                               0, 0, 0,
                                                               0, 0, 0,
                                                               /// 4
                                                               0, 0, 0,
                                                               0, 0, 0,
                                                               // ---------------------
                                                               // 5
                                                               0, 0, 0,
                                                               0, 0, 0,
                                                               // 6
                                                               0, 0, 0,
                                                               0, 0, 0}}) // rank 3
            .indices({{2, 1}, integer_type, std::vector<I> {0, 1}}) // rank 1
            .updates({{3, 3, 2, 2, 2}, numeric_type, std::vector<N> { // -------
                                                                     1, 2,
                                                                     3, 4,
                                                                     //
                                                                     5, 6,
                                                                     7, 8,
                                                                     // ---------
                                                                     //
                                                                     9, 10,
                                                                     11, 12,
                                                                     //
                                                                     13, 14,
                                                                     15, 16,
                                                                     // ------------
                                                                     //
                                                                     17, 18,
                                                                     19, 20,
                                                                     //
                                                                     21, 22,
                                                                     23, 24,
                                                                     // -------------
                                                                     //
                                                                     // ===================
                                                                     25, 26,
                                                                     27, 28,
                                                                     //
                                                                     29, 30,
                                                                     31, 32,
                                                                     // ------
                                                                     //
                                                                     33, 34,
                                                                     35, 36,
                                                                     //
                                                                     37, 38,
                                                                     39, 40,
                                                                     // --------------
                                                                     //
                                                                     41, 42,
                                                                     43, 44,
                                                                     //
                                                                     45, 46,
                                                                     47, 48,
                                                                     // -------------
                                                                     //
                                                                     // ===================
                                                                     49, 50,
                                                                     51, 52,
                                                                     //
                                                                     53, 54,
                                                                     55, 56,
                                                                     // -----------
                                                                     //
                                                                     57, 58,
                                                                     59, 60,
                                                                     //
                                                                     61, 62,
                                                                     63, 64,
                                                                     // ------------
                                                                     //
                                                                     65, 66,
                                                                     67, 68,
                                                                     //
                                                                     69, 70,
                                                                     71, 72}}) // rank 3
            .axis({{1}, integer_type, std::vector<I> {2}})
            .expected({{3, 2, 2, 3}, numeric_type, std::vector<N> {// ---------------------
                                                               // 1
                                                               1, 2, 9,
                                                               3, 4, 11,
                                                               // 2
                                                               10, 17, 18,
                                                               12, 19, 20,
                                                               // ---------------------
                                                               // 3
                                                               25, 26, 33,
                                                               27, 28, 35,
                                                               /// 4
                                                               34, 41, 42,
                                                               36, 43, 44,
                                                               // ---------------------
                                                               // 5
                                                               49, 50, 57,
                                                               51, 52, 59,
                                                               // 6
                                                               58, 65, 66,
                                                               60, 67, 68}}),
        Builder {}
            .data({{3, 3}, numeric_type, std::vector<N> {0, 0, 0,
                                                         0, 0, 0,
                                                         0, 0, 0}}) // rank 3
            .indices({{2}, integer_type, std::vector<I> {1, 2}}) // rank 1
            .updates({{3, 2}, numeric_type, std::vector<N> {1, 1,
                                                            1, 2,
                                                            2, 2}}) // rank 3
            .axis({{1}, integer_type, std::vector<I> {1}})
            .expected({{3, 3}, numeric_type, std::vector<N> {0, 1, 1,
                                                             0, 1, 2,
                                                             0, 2, 2}}),
        Builder {}
            .data({{3, 3}, numeric_type, std::vector<N> {0, 0, 0,
                                                         0, 0, 0,
                                                         0, 0, 0}}) // rank 3
            .indices({{2}, integer_type, std::vector<I> {1, 2}}) // rank 1
            .updates({{2, 3}, numeric_type, std::vector<N> {1, 1, 1,
                                                            2, 2, 2}}) // rank 3
            .axis({{1}, integer_type, std::vector<I> {0}})
            .expected({{3, 3}, numeric_type, std::vector<N> {0, 0, 0,
                                                             1, 1, 1,
                                                             2, 2, 2}}),
        Builder {}
            .data({{3, 4}, numeric_type, std::vector<N> {0, 0, 0, 0,
                                                         0, 0, 0, 0,
                                                         0, 0, 0, 0}}) // rank 3
            .indices({{2}, integer_type, std::vector<I> {0, 2}}) // rank 1
            .updates({{3, 4}, numeric_type, std::vector<N> {1, 2, 3, 7,
                                                            4, 5, 6, 8,
                                                            7, 8, 9, 10}}) // rank 3
            .axis({{1}, integer_type, std::vector<I> {0}})
            .expected({{3, 4}, numeric_type, std::vector<N> {1, 2, 3, 7,
                                                             0, 0, 0, 0,
                                                             4, 5, 6, 8}}),
        Builder {}
            .data({{3, 3}, numeric_type, std::vector<N> {0, 0, 0,
                                                         0, 0, 0,
                                                         0, 0, 0}}) // rank 3
            .indices({{2}, integer_type, std::vector<I> {0, 2}}) // rank 1
            .updates({{3, 5}, numeric_type, std::vector<N> {1, 2, 3, 4, 5,
                                                            6, 7, 8, 9, 10,
                                                            11, 12, 13, 14, 15}}) // rank 3
            .axis({{1}, integer_type, std::vector<I> {1}})
            .expected({{3, 3}, numeric_type, std::vector<N> {1, 0, 2,
                                                             6, 0, 7,
                                                             11, 0, 12}}),
        Builder {}
            .data({{3, 3}, numeric_type, std::vector<N> {0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0}}) // rank 3
            .indices({{1, 2}, integer_type, std::vector<I> {1, 2}}) // rank 1
            .updates({{1, 2, 3}, numeric_type, std::vector<N> {1, 2, 3,
                                                               4, 5, 6}}) // rank 3
            .axis({{1}, integer_type, std::vector<I> {0}})
            .expected({{3, 3}, numeric_type, std::vector<N> {0, 0, 0,
                                                             1, 2, 3,
                                                             4, 5, 6}}),
        Builder {}
            .data({{3, 3}, numeric_type, std::vector<N> {0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0}}) // rank 3
            .indices({{1, 2}, integer_type, std::vector<I> {1, 2}}) // rank 1
            .updates({{3, 1, 2}, numeric_type, std::vector<N> {1, 2,
                                                               3, 4,
                                                               5, 6}}) // rank 3
            .axis({{1}, integer_type, std::vector<I> {1}})
            .expected({{3, 3}, numeric_type, std::vector<N> {0, 1, 2,
                                                             0, 3, 4,
                                                             0, 5, 6}}),
        Builder {}
            .data({{3, 3, 2}, numeric_type, std::vector<N> {0, 0,
                                                            0, 0,
                                                            0, 0,
                                                            // --------
                                                            0, 0,
                                                            0, 0,
                                                            0, 0,
                                                            // --------
                                                            0, 0,
                                                            0, 0,
                                                            0, 0}}) // rank 3
            .indices({{1, 1}, integer_type, std::vector<I> {1}}) // rank 1
            .updates({{1, 1, 3, 2}, numeric_type, std::vector<N> {1, 2,
                                                                  3, 4,
                                                                  5, 6}}) // rank 3
            .axis({{1}, integer_type, std::vector<I> {1}})
            .expected({{3, 3, 2}, numeric_type, std::vector<N> {0, 0,
                                                                1, 2,
                                                                0, 0,
                                                                // ------
                                                                0, 0,
                                                                0, 0,
                                                                0, 0,
                                                                // -----------
                                                                0, 0,
                                                                0, 0,
                                                                0, 0}})};
    return ScatterUpdateParams;
}

std::vector<ScatterUpdate6Params> generateScatterUpdateCombinedParams() {
    const std::vector<std::vector<ScatterUpdate6Params>> ScatterUpdateTypeParams {
        // f32
        generateScatterUpdate6Params<element::Type_t::f32, element::Type_t::i16>(element::f32, element::i16),
        generateScatterUpdate6Params<element::Type_t::f32, element::Type_t::i32>(element::f32, element::i32),
        generateScatterUpdate6Params<element::Type_t::f32, element::Type_t::i64>(element::f32, element::i64),
        generateScatterUpdate6Params<element::Type_t::f32, element::Type_t::u8>(element::f32, element::u8),
        generateScatterUpdate6Params<element::Type_t::f32, element::Type_t::u16>(element::f32, element::u16),
        generateScatterUpdate6Params<element::Type_t::f32, element::Type_t::u32>(element::f32, element::u32),
        // f16
        generateScatterUpdate6Params<element::Type_t::f16, element::Type_t::i16>(element::f16, element::i16),
        generateScatterUpdate6Params<element::Type_t::f16, element::Type_t::i32>(element::f16, element::i32),
        generateScatterUpdate6Params<element::Type_t::f16, element::Type_t::i64>(element::f16, element::i64),
        generateScatterUpdate6Params<element::Type_t::f16, element::Type_t::u8>(element::f16, element::u8),
        generateScatterUpdate6Params<element::Type_t::f16, element::Type_t::u16>(element::f16, element::u16),
        generateScatterUpdate6Params<element::Type_t::f16, element::Type_t::u32>(element::f16, element::u32),
        // i8
        generateScatterUpdate6Params<element::Type_t::i8, element::Type_t::i16>(element::i8, element::i16),
        generateScatterUpdate6Params<element::Type_t::i8, element::Type_t::i32>(element::i8, element::i32),
        generateScatterUpdate6Params<element::Type_t::i8, element::Type_t::i64>(element::i8, element::i64),
        generateScatterUpdate6Params<element::Type_t::i8, element::Type_t::u8>(element::i8, element::u8),
        generateScatterUpdate6Params<element::Type_t::i8, element::Type_t::u16>(element::i8, element::u16),
        generateScatterUpdate6Params<element::Type_t::i8, element::Type_t::u32>(element::i8, element::u32),
        // i16
        generateScatterUpdate6Params<element::Type_t::i16, element::Type_t::i16>(element::i16, element::i16),
        generateScatterUpdate6Params<element::Type_t::i16, element::Type_t::i32>(element::i16, element::i32),
        generateScatterUpdate6Params<element::Type_t::i16, element::Type_t::i64>(element::i16, element::i64),
        generateScatterUpdate6Params<element::Type_t::i16, element::Type_t::u8>(element::i16, element::u8),
        generateScatterUpdate6Params<element::Type_t::i16, element::Type_t::u16>(element::i16, element::u16),
        generateScatterUpdate6Params<element::Type_t::i16, element::Type_t::u32>(element::i16, element::u32),
        // i32
        generateScatterUpdate6Params<element::Type_t::i32, element::Type_t::i16>(element::i32, element::i16),
        generateScatterUpdate6Params<element::Type_t::i32, element::Type_t::i32>(element::i32, element::i32),
        generateScatterUpdate6Params<element::Type_t::i32, element::Type_t::i64>(element::i32, element::i64),
        generateScatterUpdate6Params<element::Type_t::i32, element::Type_t::u8>(element::i32, element::u8),
        generateScatterUpdate6Params<element::Type_t::i32, element::Type_t::u16>(element::i32, element::u16),
        generateScatterUpdate6Params<element::Type_t::i32, element::Type_t::u32>(element::i32, element::u32),
        // i64
        generateScatterUpdate6Params<element::Type_t::i64, element::Type_t::i16>(element::i64, element::i16),
        generateScatterUpdate6Params<element::Type_t::i64, element::Type_t::i32>(element::i64, element::i32),
        generateScatterUpdate6Params<element::Type_t::i64, element::Type_t::i64>(element::i64, element::i64),
        generateScatterUpdate6Params<element::Type_t::i64, element::Type_t::u8>(element::i64, element::u8),
        generateScatterUpdate6Params<element::Type_t::i64, element::Type_t::u16>(element::i64, element::u16),
        generateScatterUpdate6Params<element::Type_t::i64, element::Type_t::u32>(element::i64, element::u32),
        // u8
        generateScatterUpdate6Params<element::Type_t::u8, element::Type_t::i16>(element::u8, element::i16),
        generateScatterUpdate6Params<element::Type_t::u8, element::Type_t::i32>(element::u8, element::i32),
        generateScatterUpdate6Params<element::Type_t::u8, element::Type_t::i64>(element::u8, element::i64),
        generateScatterUpdate6Params<element::Type_t::u8, element::Type_t::u8>(element::u8, element::u8),
        generateScatterUpdate6Params<element::Type_t::u8, element::Type_t::u16>(element::u8, element::u16),
        generateScatterUpdate6Params<element::Type_t::u8, element::Type_t::u32>(element::u8, element::u32),
        // u16
        generateScatterUpdate6Params<element::Type_t::u16, element::Type_t::i16>(element::u16, element::i16),
        generateScatterUpdate6Params<element::Type_t::u16, element::Type_t::i32>(element::u16, element::i32),
        generateScatterUpdate6Params<element::Type_t::u16, element::Type_t::i64>(element::u16, element::i64),
        generateScatterUpdate6Params<element::Type_t::u16, element::Type_t::u8>(element::u16, element::u8),
        generateScatterUpdate6Params<element::Type_t::u16, element::Type_t::u16>(element::u16, element::u16),
        generateScatterUpdate6Params<element::Type_t::u16, element::Type_t::u32>(element::u16, element::u32),
        // u32
        generateScatterUpdate6Params<element::Type_t::u32, element::Type_t::i16>(element::u32, element::i16),
        generateScatterUpdate6Params<element::Type_t::u32, element::Type_t::i32>(element::u32, element::i32),
        generateScatterUpdate6Params<element::Type_t::u32, element::Type_t::i64>(element::u32, element::i64),
        generateScatterUpdate6Params<element::Type_t::u32, element::Type_t::u8>(element::u32, element::u8),
        generateScatterUpdate6Params<element::Type_t::u32, element::Type_t::u16>(element::u32, element::u16),
        generateScatterUpdate6Params<element::Type_t::u32, element::Type_t::u32>(element::u32, element::u32),
        // u64
        generateScatterUpdate6Params<element::Type_t::u64, element::Type_t::i16>(element::u64, element::i16),
        generateScatterUpdate6Params<element::Type_t::u64, element::Type_t::i32>(element::u64, element::i32),
        generateScatterUpdate6Params<element::Type_t::u64, element::Type_t::i64>(element::u64, element::i64),
        generateScatterUpdate6Params<element::Type_t::u64, element::Type_t::u8>(element::u64, element::u8),
        generateScatterUpdate6Params<element::Type_t::u64, element::Type_t::u16>(element::u64, element::u16),
        generateScatterUpdate6Params<element::Type_t::u64, element::Type_t::u32>(element::u64, element::u32),
        // // bf16
        generateScatterUpdate6Params<element::Type_t::bf16, element::Type_t::i16>(element::bf16, element::i16),
        generateScatterUpdate6Params<element::Type_t::bf16, element::Type_t::i32>(element::bf16, element::i32),
        generateScatterUpdate6Params<element::Type_t::bf16, element::Type_t::i64>(element::bf16, element::i64),
        generateScatterUpdate6Params<element::Type_t::bf16, element::Type_t::u8>(element::bf16, element::u8),
        generateScatterUpdate6Params<element::Type_t::bf16, element::Type_t::u16>(element::bf16, element::u16),
        generateScatterUpdate6Params<element::Type_t::bf16, element::Type_t::u32>(element::bf16, element::u32)};
    std::vector<ScatterUpdate6Params> combinedParams;

    for (const auto& params : ScatterUpdateTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}
} // namespace

INSTANTIATE_TEST_SUITE_P(smoke_ScatterUpdate_With_Hardcoded_Refs, ReferenceScatterUpdate6LayerTest, ::testing::ValuesIn(generateScatterUpdateCombinedParams()),
                         ReferenceScatterUpdate6LayerTest::getTestCaseName);
} // namespace reference_tests
