// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_update.hpp"

#include <vector>

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;

namespace reference_tests {

namespace {

// ---------------------- V3 ------------------------------

struct ScatterUpdate3Params {
    reference_tests::Tensor data;
    reference_tests::Tensor indices;
    reference_tests::Tensor updates;
    reference_tests::Tensor axis;
    reference_tests::Tensor expected;
};

struct Builder : ParamsBuilder<ScatterUpdate3Params> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, data);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, indices);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, updates);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, axis);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceScatterUpdate6LayerTest : public testing::TestWithParam<ScatterUpdate3Params>,
                                         public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.data.data, params.indices.data, params.updates.data, params.axis.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ScatterUpdate3Params>& obj) {
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
    static std::shared_ptr<ov::Model> CreateFunction(const ScatterUpdate3Params& params) {
        const auto data_shape = params.data.shape;
        const auto indices_shape = params.indices.shape;
        const auto updates_shape = params.updates.shape;
        const auto axis_shape = params.axis.shape;
        const auto numeric_type = params.data.type;
        const auto indices_type = params.indices.type;
        const auto axis_type = params.axis.type;

        const auto data = std::make_shared<ov::op::v0::Parameter>(numeric_type, data_shape);
        const auto indices = std::make_shared<ov::op::v0::Parameter>(indices_type, indices_shape);
        const auto updates = std::make_shared<ov::op::v0::Parameter>(numeric_type, updates_shape);
        const auto axis = std::make_shared<ov::op::v0::Parameter>(axis_type, axis_shape);
        const auto scatter_update = std::make_shared<ov::op::v3::ScatterUpdate>(data, indices, updates, axis);
        return std::make_shared<ov::Model>(ov::NodeVector{scatter_update},
                                           ov::ParameterVector{data, indices, updates, axis});
    }
};

TEST_P(ReferenceScatterUpdate6LayerTest, ScatterUpdateWithHardcodedRefs) {
    Exec();
}

template <ov::element::Type_t NUM_ET, ov::element::Type_t INT_ET>
std::vector<ScatterUpdate3Params> generateScatterUpdate3Params(const ov::element::Type& numeric_type,
                                                               const ov::element::Type& integer_type) {
    using N = typename ov::element_type_traits<NUM_ET>::value_type;
    using I = typename ov::element_type_traits<INT_ET>::value_type;
    std::vector<ScatterUpdate3Params> ScatterUpdateParams{
        Builder{}
            .data({{3, 2, 2, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{2, 1}, integer_type, std::vector<I>{0, 1}})
            .updates({{3, 3, 2, 2, 2},
                      numeric_type,
                      std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                     19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                     37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                                     55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72}})
            .axis({{1}, integer_type, std::vector<I>{2}})
            .expected({{3, 2, 2, 3}, numeric_type, std::vector<N>{1,  2,  9,  3,  4,  11, 10, 17, 18, 12, 19, 20,
                                                                  25, 26, 33, 27, 28, 35, 34, 41, 42, 36, 43, 44,
                                                                  49, 50, 57, 51, 52, 59, 58, 65, 66, 60, 67, 68}}),
        Builder{}
            .data({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{2}, integer_type, std::vector<I>{1, 2}})
            .updates({{3, 2}, numeric_type, std::vector<N>{1, 1, 1, 2, 2, 2}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{3, 3}, numeric_type, std::vector<N>{0, 1, 1, 0, 1, 2, 0, 2, 2}}),
        Builder{}
            .data({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{2}, integer_type, std::vector<I>{1, 2}})
            .updates({{2, 3}, numeric_type, std::vector<N>{1, 1, 1, 2, 2, 2}})
            .axis({{1}, integer_type, std::vector<I>{0}})
            .expected({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 1, 1, 1, 2, 2, 2}}),
        Builder{}
            .data({{3, 4}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{2}, integer_type, std::vector<I>{0, 2}})
            .updates({{3, 4}, numeric_type, std::vector<N>{1, 2, 3, 7, 4, 5, 6, 8, 7, 8, 9, 10}})
            .axis({{1}, integer_type, std::vector<I>{0}})
            .expected({{3, 4}, numeric_type, std::vector<N>{1, 2, 3, 7, 0, 0, 0, 0, 4, 5, 6, 8}}),
        Builder{}
            .data({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{2}, integer_type, std::vector<I>{0, 2}})
            .updates({{3, 5}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{3, 3}, numeric_type, std::vector<N>{1, 0, 2, 6, 0, 7, 11, 0, 12}}),
        Builder{}
            .data({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 2}, integer_type, std::vector<I>{1, 2}})
            .updates({{1, 2, 3}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6}})
            .axis({{1}, integer_type, std::vector<I>{0}})
            .expected({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 1, 2, 3, 4, 5, 6}}),
        Builder{}
            .data({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 2}, integer_type, std::vector<I>{1, 2}})
            .updates({{3, 1, 2}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{3, 3}, numeric_type, std::vector<N>{0, 1, 2, 0, 3, 4, 0, 5, 6}}),
        Builder{}
            .data({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 2}, integer_type, std::vector<I>{1, 2}})
            .updates({{4, 4, 4}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                                              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                              27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                                              40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                                              53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{3, 3}, numeric_type, std::vector<N>{0, 1, 2, 0, 17, 18, 0, 33, 34}}),
        Builder{}
            .data({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 3}, integer_type, std::vector<I>{0, 1, 2}})
            .updates({{4, 4, 4}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                                              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                              27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                                              40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                                              53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{3, 3}, numeric_type, std::vector<N>{1, 2, 3, 17, 18, 19, 33, 34, 35}}),
        Builder{}
            .data({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 1}, integer_type, std::vector<I>{2}})
            .updates({{2, 2, 2}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6, 7, 8}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{3, 3}, numeric_type, std::vector<N>{0, 0, 1, 0, 0, 5, 0, 0, 0}}),
        Builder{}
            .data({{3, 4}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 4}, integer_type, std::vector<I>{0, 1, 2, 3}})
            .updates({{2, 2, 2}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6, 7, 8}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{3, 4}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0}}),
        Builder{}
            .data({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 3}, integer_type, std::vector<I>{0, 1, 2}})
            .updates({{2, 2, 2}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6, 7, 8}})
            .axis({{1}, integer_type, std::vector<I>{0}})
            .expected({{3, 3}, numeric_type, std::vector<N>{1, 2, 0, 3, 4, 0, 5, 6, 0}}),
        Builder{}
            .data({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 3}, integer_type, std::vector<I>{0, 1, 2}})
            .updates({{2, 2, 1}, numeric_type, std::vector<N>{1, 2, 3, 4}})
            .axis({{1}, integer_type, std::vector<I>{0}})
            .expected({{3, 3}, numeric_type, std::vector<N>{1, 0, 0, 2, 0, 0, 3, 0, 0}}),
        Builder{}
            .data({{3, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 3}, integer_type, std::vector<I>{0, 1, 2}})
            .updates({{1, 1, 1}, numeric_type, std::vector<N>{1}})
            .axis({{1}, integer_type, std::vector<I>{0}})
            .expected({{3, 3}, numeric_type, std::vector<N>{1, 0, 0, 0, 0, 0, 0, 0, 0}}),
        Builder{}
            .data({{2, 2}, numeric_type, std::vector<N>{0, 0, 0, 0}})
            .indices({{2, 1}, integer_type, std::vector<I>{0, 1}})
            .updates({{2, 2, 2}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6, 7, 8}})
            .axis({{1}, integer_type, std::vector<I>{0}})
            .expected({{2, 2}, numeric_type, std::vector<N>{1, 2, 3, 4}}),
        Builder{}
            .data({{4, 4}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{4, 1}, integer_type, std::vector<I>{0, 1, 2, 3}})
            .updates({{2, 2, 2}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6, 7, 8}})
            .axis({{1}, integer_type, std::vector<I>{0}})
            .expected({{4, 4}, numeric_type, std::vector<N>{1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0}}),
        Builder{}
            .data({{2, 3, 4, 2}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{3, 1}, integer_type, std::vector<I>{0, 1, 2}})
            .updates({{3, 2, 3, 3, 2},
                      numeric_type,
                      std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  17,  18,
                                     19, 20, 21, 22, 23, 24, 25, 26, 27, 28,  29,  30,  31,  32,  33,  34,  35,  36,
                                     37, 38, 39, 40, 41, 42, 43, 44, 45, 46,  47,  48,  49,  50,  51,  52,  53,  54,
                                     55, 56, 57, 58, 59, 60, 61, 62, 63, 64,  65,  66,  67,  68,  69,  70,  71,  72,
                                     73, 74, 75, 76, 77, 78, 79, 80, 81, 82,  83,  84,  85,  86,  87,  88,  89,  90,
                                     91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108}})
            .axis({{1}, integer_type, std::vector<I>{2}})
            .expected({{2, 3, 4, 2}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  0,  0,  19, 20, 21, 22,
                                                                  23, 24, 0,  0,  37, 38, 39, 40, 41, 42, 0,  0,
                                                                  55, 56, 57, 58, 59, 60, 0,  0,  73, 74, 75, 76,
                                                                  77, 78, 0,  0,  91, 92, 93, 94, 95, 96, 0,  0}}),
        Builder{}
            .data({{1, 3, 2, 2}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 3}, integer_type, std::vector<I>{2, 0, 1}})
            .updates({{1, 3, 2, 2, 2}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{1, 3, 2, 2}, numeric_type, std::vector<N>{5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4}}),
        Builder{}
            .data({{2, 2, 2}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 2}, integer_type, std::vector<I>{0, 1}})
            .updates({{2, 2, 3, 2}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{2, 2, 2}, numeric_type, std::vector<N>{1, 2, 3, 4, 13, 14, 15, 16}}),
        Builder{}
            .data({{2, 2, 4}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 2, 1}, integer_type, std::vector<I>{0, 1}})
            .updates({{2, 2, 3, 2, 1}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{2, 2, 4}, numeric_type, std::vector<N>{1, 13, 0, 0, 2, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}),
        Builder{}
            .data({{2, 4, 2}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 2, 1}, integer_type, std::vector<I>{0, 1}})
            .updates({{2, 2, 3, 2, 1}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{2, 4, 2}, numeric_type, std::vector<N>{1, 13, 2, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}),
        Builder{}
            .data({{2, 2, 2}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{2, 1}, integer_type, std::vector<I>{1, 0}})
            .updates({{2, 2, 3, 2}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}})
            .axis({{1}, integer_type, std::vector<I>{2}})
            .expected({{2, 2, 2},
                       numeric_type,
                       std::vector<N>{2,
                                      1,
                                      8,
                                      7,
                                      //
                                      14,
                                      13,
                                      20,
                                      19}}),
        Builder{}
            .data({{2, 2, 4}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 2, 1}, integer_type, std::vector<I>{0, 1}})
            .updates({{2, 2, 3, 1, 2}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}})
            .axis({{1}, integer_type, std::vector<I>{1}})
            .expected({{2, 2, 4}, numeric_type, std::vector<N>{1, 2, 13, 14, 3, 4, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0}}),

        Builder{}
            .data({{3, 2, 2, 2}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{2}, integer_type, std::vector<I>{0, 1}})
            .updates(
                {{2, 2, 2, 2}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}})
            .axis({{1}, integer_type, std::vector<I>{3}})
            .expected({{3, 2, 2, 2}, numeric_type, std::vector<N>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12,
                                                                  13, 14, 15, 16, 0, 0, 0, 0, 0, 0,  0,  0}}),
        Builder{}
            .data({{5, 2, 2, 2}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{2}, integer_type, std::vector<I>{0, 1}})
            .updates(
                {{2, 2, 2, 2}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}})
            .axis({{1}, integer_type, std::vector<I>{2}})
            .expected({{5, 2, 2, 2}, numeric_type, std::vector<N>{1,  2,  3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                                  15, 16, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,
                                                                  0,  0,  0, 0, 0, 0, 0, 0, 0, 0,  0,  0}}),
        Builder{}
            .data({{5, 2, 2, 2}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{2, 1}, integer_type, std::vector<I>{0, 1}})
            .updates({{2, 2, 2, 2, 2}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32}})
            .axis({{1}, integer_type, std::vector<I>{2}})
            .expected({{5, 2, 2, 2}, numeric_type, std::vector<N>{1,  2,  3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 25, 26,
                                                                  27, 28, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,
                                                                  0,  0,  0, 0, 0, 0,  0,  0,  0,  0,  0,  0}})};
    return ScatterUpdateParams;
}

template <ov::element::Type_t NUM_ET, ov::element::Type_t INT_ET>
std::vector<ScatterUpdate3Params> generateScatterUpdate3ParamsNegativeAxis(const ov::element::Type& numeric_type,
                                                                           const ov::element::Type& integer_type) {
    using N = typename ov::element_type_traits<NUM_ET>::value_type;
    using I = typename ov::element_type_traits<INT_ET>::value_type;
    std::vector<ScatterUpdate3Params> ScatterUpdateParams{
        Builder{}
            .data({{2, 2, 3}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 2, 1}, integer_type, std::vector<I>{0, 1}})
            .updates({{2, 2, 3, 1, 2}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}})
            .axis({{1}, integer_type, std::vector<I>{-2}})
            .expected({{2, 2, 3}, numeric_type, std::vector<N>{1, 2, 13, 3, 4, 15, 14, 0, 0, 16, 0, 0}}),
        Builder{}
            .data({{2, 2, 2}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{1, 2, 1}, integer_type, std::vector<I>{0, 1}})
            .updates({{2, 2, 3, 1, 2}, numeric_type, std::vector<N>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}})
            .axis({{1}, integer_type, std::vector<I>{-1}})
            .expected({{2, 2, 2}, numeric_type, std::vector<N>{1, 2, 7, 8, 13, 14, 19, 20}}),
        Builder{}
            .data({{4, 2, 2, 2}, numeric_type, std::vector<N>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
            .indices({{2}, integer_type, std::vector<I>{0, 1}})
            .updates(
                {{2, 2, 2, 2}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}})
            .axis({{1}, integer_type, std::vector<I>{-3}})
            .expected(
                {{4, 2, 2, 2}, numeric_type, std::vector<N>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0}})};
    return ScatterUpdateParams;
}

std::vector<ScatterUpdate3Params> generateScatterUpdateCombinedParams() {
    const std::vector<std::vector<ScatterUpdate3Params>> ScatterUpdateTypeParams{
        // f32
        generateScatterUpdate3Params<ov::element::Type_t::f32, ov::element::Type_t::i16>(ov::element::f32,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::f32, ov::element::Type_t::i32>(ov::element::f32,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::f32, ov::element::Type_t::i64>(ov::element::f32,
                                                                                         ov::element::i64),
        generateScatterUpdate3Params<ov::element::Type_t::f32, ov::element::Type_t::u32>(ov::element::f32,
                                                                                         ov::element::u32),
        generateScatterUpdate3Params<ov::element::Type_t::f32, ov::element::Type_t::u64>(ov::element::f32,
                                                                                         ov::element::u64),

        // f16
        generateScatterUpdate3Params<ov::element::Type_t::f16, ov::element::Type_t::i16>(ov::element::f16,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::f16, ov::element::Type_t::i32>(ov::element::f16,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::f16, ov::element::Type_t::i64>(ov::element::f16,
                                                                                         ov::element::i64),
        generateScatterUpdate3Params<ov::element::Type_t::f16, ov::element::Type_t::u32>(ov::element::f16,
                                                                                         ov::element::u32),
        generateScatterUpdate3Params<ov::element::Type_t::f16, ov::element::Type_t::u64>(ov::element::f16,
                                                                                         ov::element::u64),
        // i8
        generateScatterUpdate3Params<ov::element::Type_t::i8, ov::element::Type_t::i16>(ov::element::i8,
                                                                                        ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::i8, ov::element::Type_t::i32>(ov::element::i8,
                                                                                        ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::i8, ov::element::Type_t::i64>(ov::element::i8,
                                                                                        ov::element::i64),
        generateScatterUpdate3Params<ov::element::Type_t::i8, ov::element::Type_t::u32>(ov::element::i8,
                                                                                        ov::element::u32),
        generateScatterUpdate3Params<ov::element::Type_t::i8, ov::element::Type_t::u64>(ov::element::i8,
                                                                                        ov::element::u64),
        // i16
        generateScatterUpdate3Params<ov::element::Type_t::i16, ov::element::Type_t::i16>(ov::element::i16,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::i16, ov::element::Type_t::i32>(ov::element::i16,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::i16, ov::element::Type_t::i64>(ov::element::i16,
                                                                                         ov::element::i64),
        generateScatterUpdate3Params<ov::element::Type_t::i16, ov::element::Type_t::u32>(ov::element::i16,
                                                                                         ov::element::u32),
        generateScatterUpdate3Params<ov::element::Type_t::i16, ov::element::Type_t::u64>(ov::element::i16,
                                                                                         ov::element::u64),
        // i32
        generateScatterUpdate3Params<ov::element::Type_t::i32, ov::element::Type_t::i16>(ov::element::i32,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::i32, ov::element::Type_t::i32>(ov::element::i32,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::i32, ov::element::Type_t::i64>(ov::element::i32,
                                                                                         ov::element::i64),
        generateScatterUpdate3Params<ov::element::Type_t::i32, ov::element::Type_t::u32>(ov::element::i32,
                                                                                         ov::element::u32),
        generateScatterUpdate3Params<ov::element::Type_t::i32, ov::element::Type_t::u64>(ov::element::i32,
                                                                                         ov::element::u64),
        // i64
        generateScatterUpdate3Params<ov::element::Type_t::i64, ov::element::Type_t::i16>(ov::element::i64,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::i64, ov::element::Type_t::i32>(ov::element::i64,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::i64, ov::element::Type_t::i64>(ov::element::i64,
                                                                                         ov::element::i64),
        generateScatterUpdate3Params<ov::element::Type_t::i64, ov::element::Type_t::u32>(ov::element::i64,
                                                                                         ov::element::u32),
        generateScatterUpdate3Params<ov::element::Type_t::i64, ov::element::Type_t::u64>(ov::element::i64,
                                                                                         ov::element::u64),
        // u8
        generateScatterUpdate3Params<ov::element::Type_t::u8, ov::element::Type_t::i16>(ov::element::u8,
                                                                                        ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::u8, ov::element::Type_t::i32>(ov::element::u8,
                                                                                        ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::u8, ov::element::Type_t::i64>(ov::element::u8,
                                                                                        ov::element::i64),
        generateScatterUpdate3Params<ov::element::Type_t::u8, ov::element::Type_t::u32>(ov::element::u8,
                                                                                        ov::element::u32),
        generateScatterUpdate3Params<ov::element::Type_t::u8, ov::element::Type_t::u64>(ov::element::u8,
                                                                                        ov::element::u64),
        // u16
        generateScatterUpdate3Params<ov::element::Type_t::u16, ov::element::Type_t::i16>(ov::element::u16,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::u16, ov::element::Type_t::i32>(ov::element::u16,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::u16, ov::element::Type_t::i64>(ov::element::u16,
                                                                                         ov::element::i64),
        generateScatterUpdate3Params<ov::element::Type_t::u16, ov::element::Type_t::u32>(ov::element::u16,
                                                                                         ov::element::u32),
        generateScatterUpdate3Params<ov::element::Type_t::u16, ov::element::Type_t::u64>(ov::element::u16,
                                                                                         ov::element::u64),
        // u32
        generateScatterUpdate3Params<ov::element::Type_t::u32, ov::element::Type_t::i16>(ov::element::u32,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::u32, ov::element::Type_t::i32>(ov::element::u32,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::u32, ov::element::Type_t::i64>(ov::element::u32,
                                                                                         ov::element::i64),
        generateScatterUpdate3Params<ov::element::Type_t::u32, ov::element::Type_t::u32>(ov::element::u32,
                                                                                         ov::element::u32),
        generateScatterUpdate3Params<ov::element::Type_t::u32, ov::element::Type_t::u64>(ov::element::u32,
                                                                                         ov::element::u64),
        // u64
        generateScatterUpdate3Params<ov::element::Type_t::u64, ov::element::Type_t::i16>(ov::element::u64,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::u64, ov::element::Type_t::i32>(ov::element::u64,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::u64, ov::element::Type_t::i64>(ov::element::u64,
                                                                                         ov::element::i64),
        generateScatterUpdate3Params<ov::element::Type_t::u64, ov::element::Type_t::u32>(ov::element::u64,
                                                                                         ov::element::u32),
        generateScatterUpdate3Params<ov::element::Type_t::u64, ov::element::Type_t::u64>(ov::element::u64,
                                                                                         ov::element::u64),
        // bf16
        generateScatterUpdate3Params<ov::element::Type_t::bf16, ov::element::Type_t::i16>(ov::element::bf16,
                                                                                          ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::bf16, ov::element::Type_t::i32>(ov::element::bf16,
                                                                                          ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::bf16, ov::element::Type_t::i64>(ov::element::bf16,
                                                                                          ov::element::i64),
        generateScatterUpdate3Params<ov::element::Type_t::bf16, ov::element::Type_t::u32>(ov::element::bf16,
                                                                                          ov::element::u32),
        generateScatterUpdate3Params<ov::element::Type_t::bf16, ov::element::Type_t::u64>(ov::element::bf16,
                                                                                          ov::element::u64)};
    std::vector<ScatterUpdate3Params> combinedParams;

    for (const auto& params : ScatterUpdateTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

std::vector<ScatterUpdate3Params> generateScatterUpdateNegativeAxisParams() {
    const std::vector<std::vector<ScatterUpdate3Params>> ScatterUpdateTypeParams{
        // f32
        generateScatterUpdate3Params<ov::element::Type_t::f32, ov::element::Type_t::i16>(ov::element::f32,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::f32, ov::element::Type_t::i32>(ov::element::f32,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::f32, ov::element::Type_t::i64>(ov::element::f32,
                                                                                         ov::element::i64),
        // f16
        generateScatterUpdate3Params<ov::element::Type_t::f16, ov::element::Type_t::i16>(ov::element::f16,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::f16, ov::element::Type_t::i32>(ov::element::f16,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::f16, ov::element::Type_t::i64>(ov::element::f16,
                                                                                         ov::element::i64),
        // i8
        generateScatterUpdate3Params<ov::element::Type_t::i8, ov::element::Type_t::i16>(ov::element::i8,
                                                                                        ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::i8, ov::element::Type_t::i32>(ov::element::i8,
                                                                                        ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::i8, ov::element::Type_t::i64>(ov::element::i8,
                                                                                        ov::element::i64),
        // i16
        generateScatterUpdate3Params<ov::element::Type_t::i16, ov::element::Type_t::i16>(ov::element::i16,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::i16, ov::element::Type_t::i32>(ov::element::i16,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::i16, ov::element::Type_t::i64>(ov::element::i16,
                                                                                         ov::element::i64),
        // i32
        generateScatterUpdate3Params<ov::element::Type_t::i32, ov::element::Type_t::i16>(ov::element::i32,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::i32, ov::element::Type_t::i32>(ov::element::i32,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::i32, ov::element::Type_t::i64>(ov::element::i32,
                                                                                         ov::element::i64),
        // i64
        generateScatterUpdate3Params<ov::element::Type_t::i64, ov::element::Type_t::i16>(ov::element::i64,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::i64, ov::element::Type_t::i32>(ov::element::i64,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::i64, ov::element::Type_t::i64>(ov::element::i64,
                                                                                         ov::element::i64),
        // u8
        generateScatterUpdate3Params<ov::element::Type_t::u8, ov::element::Type_t::i16>(ov::element::u8,
                                                                                        ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::u8, ov::element::Type_t::i32>(ov::element::u8,
                                                                                        ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::u8, ov::element::Type_t::i64>(ov::element::u8,
                                                                                        ov::element::i64),
        // u16
        generateScatterUpdate3Params<ov::element::Type_t::u16, ov::element::Type_t::i16>(ov::element::u16,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::u16, ov::element::Type_t::i32>(ov::element::u16,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::u16, ov::element::Type_t::i64>(ov::element::u16,
                                                                                         ov::element::i64),
        // u32
        generateScatterUpdate3Params<ov::element::Type_t::u32, ov::element::Type_t::i16>(ov::element::u32,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::u32, ov::element::Type_t::i32>(ov::element::u32,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::u32, ov::element::Type_t::i64>(ov::element::u32,
                                                                                         ov::element::i64),
        // u64
        generateScatterUpdate3Params<ov::element::Type_t::u64, ov::element::Type_t::i16>(ov::element::u64,
                                                                                         ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::u64, ov::element::Type_t::i32>(ov::element::u64,
                                                                                         ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::u64, ov::element::Type_t::i64>(ov::element::u64,
                                                                                         ov::element::i64),
        // bf16
        generateScatterUpdate3Params<ov::element::Type_t::bf16, ov::element::Type_t::i16>(ov::element::bf16,
                                                                                          ov::element::i16),
        generateScatterUpdate3Params<ov::element::Type_t::bf16, ov::element::Type_t::i32>(ov::element::bf16,
                                                                                          ov::element::i32),
        generateScatterUpdate3Params<ov::element::Type_t::bf16, ov::element::Type_t::i64>(ov::element::bf16,
                                                                                          ov::element::i64)};
    std::vector<ScatterUpdate3Params> combinedParams;

    for (const auto& params : ScatterUpdateTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_ScatterUpdate_With_Hardcoded_Refs,
                         ReferenceScatterUpdate6LayerTest,
                         ::testing::ValuesIn(generateScatterUpdateCombinedParams()),
                         ReferenceScatterUpdate6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterUpdate_Negative_Axis_With_Hardcoded_Refs,
                         ReferenceScatterUpdate6LayerTest,
                         ::testing::ValuesIn(generateScatterUpdateNegativeAxisParams()),
                         ReferenceScatterUpdate6LayerTest::getTestCaseName);
}  // namespace reference_tests
