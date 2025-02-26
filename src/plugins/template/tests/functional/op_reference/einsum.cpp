// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/einsum.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct EinsumParams {
    std::vector<reference_tests::Tensor> inputs;
    std::string equation;
    reference_tests::Tensor expectedResult;
    std::string testcaseName;
};

struct Builder : ParamsBuilder<EinsumParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, inputs);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, equation);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expectedResult);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, testcaseName);
};

class ReferenceEinsumTest : public testing::TestWithParam<EinsumParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateModel(params);
        for (const auto& input_tensor : params.inputs) {
            inputData.push_back(input_tensor.data);
        }
        refOutData = {params.expectedResult.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<EinsumParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.inputs[0].type;
        result << "_iShape=" << param.inputs[0].shape;
        result << "_equation=" << param.equation;
        result << "_eType=" << param.expectedResult.type;
        result << "_eShape=" << param.expectedResult.shape;
        if (param.testcaseName != "") {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateModel(const EinsumParams& params) {
        OutputVector output_vector;
        ParameterVector param_vector;
        for (const auto& input_tensor : params.inputs) {
            auto param = std::make_shared<op::v0::Parameter>(input_tensor.type, input_tensor.shape);
            output_vector.push_back(param);
            param_vector.push_back(param);
        }
        const auto einsum = std::make_shared<op::v7::Einsum>(output_vector, params.equation);
        const auto f = std::make_shared<Model>(OutputVector{einsum}, param_vector);
        return f;
    }
};

TEST_P(ReferenceEinsumTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<EinsumParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<EinsumParams> params{
        Builder{}
            .inputs({{ET, {1, 2}, std::vector<T>{1, 2}},
                     {ET, {3, 4}, std::vector<T>{3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}}})
            .equation("ab,cd->abcd")
            .expectedResult({ET, {1, 2, 3, 4}, std::vector<T>{3, 4, 5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                              6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28}})
            .testcaseName("einsum_no_reduction"),
        Builder{}
            .inputs({{ET, {1, 2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}}})
            .equation("ijk->kij")
            .expectedResult({ET, {3, 1, 2}, std::vector<T>{1, 4, 2, 5, 3, 6}})
            .testcaseName("einsum_transpose"),

        Builder{}
            .inputs({{ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}}})
            .equation("ab->a")
            .expectedResult({ET, {2}, std::vector<T>{6, 15}})
            .testcaseName("einsum_reduce"),

        Builder{}
            .inputs({{ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}}, {ET, {3, 2}, std::vector<T>{1, 2, 3, 4, 5, 6}}})
            .equation("ab,bc->ac")
            .expectedResult({ET, {2, 2}, std::vector<T>{22, 28, 49, 64}})
            .testcaseName("einsum_matrix_multiplication"),

        Builder{}
            .inputs({{ET, {2, 4}, std::vector<T>{1, 3, 2, 7, 5, 6, 0, 1}},
                     {ET, {4, 3, 1}, std::vector<T>{1, 2, 3, 4, 5, 6, 5, 7, 3, 7, 9, 1}},
                     {ET, {4, 3}, std::vector<T>{4, 3, 1, 6, 4, 2, 2, 5, 3, 1, 9, 4}}})
            .equation("ab,bcd,bc->ca")
            .expectedResult({ET, {3, 2}, std::vector<T>{145, 171, 703, 231, 85, 91}})
            .testcaseName("einsum_multiple_multiplication"),

        Builder{}
            .inputs({{ET, {2, 2, 3}, std::vector<T>{1, 3, 2, 7, 5, 6, 3, 5, 2, 1, 0, 7}}})
            .equation("a...->...")
            .expectedResult({ET, {2, 3}, std::vector<T>{4, 8, 4, 8, 5, 13}})
            .testcaseName("einsum_ellipsis_one_input_reduction"),

        Builder{}
            .inputs({{ET, {2, 2, 3}, std::vector<T>{1, 3, 2, 7, 5, 6, 3, 5, 2, 1, 0, 7}}})
            .equation("a...->...a")
            .expectedResult({ET, {2, 3, 2}, std::vector<T>{1, 3, 3, 5, 2, 2, 7, 1, 5, 0, 6, 7}})
            .testcaseName("einsum_ellipsis_one_input_transpose"),

        Builder{}
            .inputs({{ET, {2, 2, 3}, std::vector<T>{1, 3, 2, 7, 5, 6, 3, 5, 2, 1, 0, 7}}, {ET, {1}, std::vector<T>{2}}})
            .equation("ab...,...->ab...")
            .expectedResult({ET, {2, 2, 3}, std::vector<T>{2, 6, 4, 14, 10, 12, 6, 10, 4, 2, 0, 14}})
            .testcaseName("einsum_ellipsis_mul_by_1dscalar"),

        Builder{}
            .inputs({{ET, {1, 1, 4, 3}, std::vector<T>{1, 3, 2, 7, 5, 6, 3, 5, 2, 1, 0, 7}},
                     {ET, {3, 4, 2, 1}, std::vector<T>{3, 1, 6, 2, 3, 10, 9,  8, 2, 9, 3, 2,
                                                       4, 2, 3, 1, 9, 1,  11, 4, 7, 2, 3, 1}}})
            .equation("a...j,j...->a...")
            .expectedResult(
                {ET, {1, 4, 2, 4}, std::vector<T>{27, 85, 37, 66, 30, 58, 50, 8,  37, 123, 55, 83, 16, 48, 24, 30,
                                                  29, 83, 43, 52, 20, 92, 44, 24, 24, 96,  48, 30, 13, 67, 31, 15}})
            .testcaseName("einsum_ellipsis_complex_mul"),

        Builder{}
            .inputs({{ET, {1, 3, 3}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9}}})
            .equation("kii->ki")
            .expectedResult({ET, {1, 3}, std::vector<T>{1, 5, 9}})
            .testcaseName("einsum_diagonal"),

        Builder{}
            .inputs({{ET, {2, 3, 3, 2, 4}, std::vector<T>{4, 2, 5, 4, 5, 5, 1, 1, 3, 3, 1, 1, 2, 2, 4, 1, 3, 4, 4, 5, 1,
                                                          3, 1, 3, 1, 4, 3, 5, 4, 4, 5, 4, 4, 5, 4, 2, 2, 2, 3, 3, 1, 1,
                                                          4, 3, 4, 2, 2, 1, 1, 2, 3, 1, 1, 4, 2, 3, 1, 3, 4, 2, 5, 5, 3,
                                                          4, 3, 4, 5, 4, 4, 5, 1, 3, 4, 4, 5, 3, 1, 3, 2, 5, 3, 2, 5, 4,
                                                          4, 2, 4, 4, 1, 4, 4, 5, 4, 4, 4, 2, 3, 3, 4, 2, 4, 2, 5, 1, 3,
                                                          2, 4, 3, 5, 1, 2, 3, 1, 1, 2, 5, 1, 1, 2, 1, 4, 5, 3, 4, 1, 3,
                                                          3, 1, 3, 2, 4, 5, 1, 1, 5, 4, 5, 2, 2, 3, 3, 1, 2, 4}},
                     {ET, {3, 2, 1}, std::vector<T>{1, 4, 4, 5, 3, 3}}})
            .equation("abbac,bad->ad")
            .expectedResult({ET, {2, 1}, std::vector<T>{123, 129}})
            .testcaseName("einsum_diagonal_with_matmul"),

        Builder{}
            .inputs({{ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}}})
            .equation("...->...")
            .expectedResult({ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}})
            .testcaseName("einsum_identity"),
        Builder{}
            .inputs({{ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}}})
            .equation("i...->i")
            .expectedResult({ET, {2}, std::vector<T>{6, 15}})
            .testcaseName("einsum_reduce_ellipsis"),
        Builder{}
            .inputs({{ET, {3, 3, 3}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}}})
            .equation("iii->")
            .expectedResult({ET, {}, std::vector<T>{42}})
            .testcaseName("einsum_trace"),
        Builder{}
            .inputs({{ET, {3, 3, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                                    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}}})
            .equation("ii...->")
            .expectedResult({ET, {}, std::vector<T>{222}})
            .testcaseName("einsum_trace_ellipsis"),
        Builder{}
            .inputs({{ET, {3, 2, 1, 2, 1, 3, 1}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                                                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}}})
            .equation("ijkjkik->ijk")
            .expectedResult({ET, {3, 2, 1}, std::vector<T>{1, 10, 14, 23, 27, 36}})
            .testcaseName("einsum_diagonal_mixed_order"),
        Builder{}
            .inputs({{ET,
                      {3, 3, 3, 3, 3},
                      std::vector<T>{
                          1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
                          20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                          39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,
                          58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,
                          77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
                          96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                          115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
                          134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
                          153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                          172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
                          191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                          210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228,
                          229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243}}})
            .equation("iiiii->i")
            .expectedResult({ET, {3}, std::vector<T>{1, 122, 243}})
            .testcaseName("einsum_5d_diagonal"),
        Builder{}
            .inputs({{ET, {2, 1}, std::vector<T>{1, 2}},
                     {ET, {4, 1, 1}, std::vector<T>{1, 2, 3, 4}},
                     {ET, {3, 1, 3}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9}}})
            .equation("ab,bcd,dbc->ca")
            .expectedResult({ET, {3, 2}, std::vector<T>{120, 240, 150, 300, 180, 360}})
            .testcaseName("einsum_3in_broadcast"),
        Builder{}
            .inputs({{ET, {2, 1}, std::vector<T>{1, 2}}, {ET, {3, 2}, std::vector<T>{1, 2, 3, 4, 5, 6}}})
            .equation("ab,bc->ac")
            .expectedResult({ET, {2, 2}, std::vector<T>{9, 12, 18, 24}})
            .testcaseName("einsum_2in_broadcast_lhs_reduced"),
        Builder{}
            .inputs({{ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}}, {ET, {1, 2}, std::vector<T>{1, 2}}})
            .equation("ab,bc->ac")
            .expectedResult({ET, {2, 2}, std::vector<T>{6, 12, 15, 30}})
            .testcaseName("einsum_2in_broadcast_rhs_reduced"),
        Builder{}
            .inputs({{ET, {2, 1}, std::vector<T>{1, 2}}, {ET, {3, 2}, std::vector<T>{1, 2, 3, 4, 5, 6}}})
            .equation("ab,bc->bc")
            .expectedResult({ET, {3, 2}, std::vector<T>{3, 6, 9, 12, 15, 18}})
            .testcaseName("einsum_2in_broadcast_lhs_common"),
        Builder{}
            .inputs({{ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}}, {ET, {1, 2}, std::vector<T>{1, 2}}})
            .equation("ab,bc->cb")
            .expectedResult({ET, {2, 3}, std::vector<T>{5, 7, 9, 10, 14, 18}})
            .testcaseName("einsum_2in_broadcast_rhs_common"),
        Builder{}
            .inputs({{ET, {1, 3}, std::vector<T>{1, 2, 3}},
                     {ET, {3, 4, 2, 1}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}}})
            .equation("aj,j...->a...")
            .expectedResult({ET, {1, 4, 2, 1}, std::vector<T>{70, 76, 82, 88, 94, 100, 106, 112}})
            .testcaseName("einsum_2in_only_rhs_out_ellipsis"),
        Builder{}
            .inputs({{ET,
                      {2, 7, 4, 3},
                      std::vector<T>{
                          1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
                          20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                          39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,
                          58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,
                          77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
                          96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                          115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
                          134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
                          153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168}},
                     {ET, {3}, std::vector<T>{1, 2, 3}}})
            .equation("a...j,j->a...")
            .expectedResult(
                {ET, {2, 7, 4}, std::vector<T>{14,  32,  50,  68,  86,  104, 122, 140, 158, 176, 194, 212, 230, 248,
                                               266, 284, 302, 320, 338, 356, 374, 392, 410, 428, 446, 464, 482, 500,
                                               518, 536, 554, 572, 590, 608, 626, 644, 662, 680, 698, 716, 734, 752,
                                               770, 788, 806, 824, 842, 860, 878, 896, 914, 932, 950, 968, 986, 1004}})
            .testcaseName("einsum_2in_only_lhs_out_ellipsis"),
        Builder{}
            .inputs({{ET,
                      {2, 7, 4, 3},
                      std::vector<T>{
                          1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
                          20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                          39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,
                          58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,
                          77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
                          96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                          115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
                          134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
                          153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168}},
                     {ET, {3}, std::vector<T>{1, 2, 3}}})
            .equation("a...j,j->a")
            .expectedResult({ET, {2}, std::vector<T>{7196, 21308}})
            .testcaseName("einsum_2in_lhs_ellipsis_out_reduced"),
        Builder{}
            .inputs({{ET, {1, 3}, std::vector<T>{1, 2, 3}},
                     {ET, {3, 4, 2, 1}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}}})
            .equation("aj,j...->a")
            .expectedResult({ET, {1}, std::vector<T>{728}})
            .testcaseName("einsum_2in_rhs_ellipsis_out_reduced"),
        Builder{}
            .inputs({{ET, {1, 1, 4, 3}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
                     {ET, {3, 4, 2, 1}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}}})
            .equation("a...j,j...->a")
            .expectedResult({ET, {1}, std::vector<T>{8312}})
            .testcaseName("einsum_2in_broadcast_ellipsis_out_reduced"),
        Builder{}
            .inputs({{ET, {1, 3}, std::vector<T>{1, 2, 3}},
                     {ET, {3, 4, 2, 1}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}}})
            .equation("a...j,j...->a...")
            .expectedResult({ET, {1, 4, 2, 1}, std::vector<T>{70, 76, 82, 88, 94, 100, 106, 112}})
            .testcaseName("einsum_2in_unsqueeze_lhs_ellipsis"),
        Builder{}
            .inputs({{ET, {1, 1, 4, 3}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
                     {ET, {3}, std::vector<T>{1, 2, 3}}})
            .equation("a...j,j...->a...")
            .expectedResult({ET, {1, 1, 4}, std::vector<T>{14, 32, 50, 68}})
            .testcaseName("einsum_2in_unsqueeze_rhs_ellipsis"),
        Builder{}
            .inputs({{ET, {1, 3}, std::vector<T>{1, 2, 3}},
                     {ET, {3, 4, 2, 1}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}}})
            .equation("a...j,j...->a")
            .expectedResult({ET, {1}, std::vector<T>{728}})
            .testcaseName("einsum_2in_unsqueeze_lhs_ellipsis_no_out_ellipsis"),
        Builder{}
            .inputs({{ET, {1, 1, 4, 3}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
                     {ET, {3}, std::vector<T>{1, 2, 3}}})
            .equation("a...j,j...->a")
            .expectedResult({ET, {1}, std::vector<T>{164}})
            .testcaseName("einsum_2in_unsqueeze_rhs_ellipsis_no_out_ellipsis"),
        Builder{}
            .inputs({{ET, {1, 3}, std::vector<T>{1, 2, 3}}, {ET, {3}, std::vector<T>{1, 2, 3}}})
            .equation("a...j,j->a...")
            .expectedResult({ET, {1}, std::vector<T>{14}})
            .testcaseName("einsum_2in_prune_lhs_out_ellipsis"),
        Builder{}
            .inputs({{ET, {1, 3}, std::vector<T>{1, 2, 3}}, {ET, {3}, std::vector<T>{1, 2, 3}}})
            .equation("aj,j...->a...")
            .expectedResult({ET, {1}, std::vector<T>{14}})
            .testcaseName("einsum_2in_prune_rhs_out_ellipsis"),
        Builder{}
            .inputs({{ET, {1, 3}, std::vector<T>{1, 2, 3}}, {ET, {3}, std::vector<T>{1, 2, 3}}})
            .equation("aj,j->a...")
            .expectedResult({ET, {1}, std::vector<T>{14}})
            .testcaseName("einsum_2in_prune_out_ellipsis"),
        Builder{}
            .inputs({{ET, {1, 3}, std::vector<T>{1, 2, 3}}, {ET, {3}, std::vector<T>{1, 2, 3}}})
            .equation("a...j,j...->a...")
            .expectedResult({ET, {1}, std::vector<T>{14}})
            .testcaseName("einsum_2in_prune_all_ellipsis"),
        Builder{}
            .inputs({{ET, {1, 3}, std::vector<T>{1, 2, 3}}, {ET, {1}, std::vector<T>{1}}})
            .equation("a...j,j->a")
            .expectedResult({ET, {1}, std::vector<T>{6}})
            .testcaseName("einsum_2in_prune_lhs_ellipsis_no_out_ellipsis"),
        Builder{}
            .inputs({{ET, {1, 1}, std::vector<T>{1}}, {ET, {3}, std::vector<T>{1, 2, 3}}})
            .equation("aj,j...->a")
            .expectedResult({ET, {1}, std::vector<T>{6}})
            .testcaseName("einsum_2in_prune_rhs_ellipsis_no_out_ellipsis"),
        Builder{}
            .inputs({{ET, {1, 3}, std::vector<T>{1, 2, 3}}, {ET, {3}, std::vector<T>{1, 2, 3}}})
            .equation("a...j,j...->a")
            .expectedResult({ET, {1}, std::vector<T>{14}})
            .testcaseName("einsum_2in_prune_inp_ellipsis_no_out_ellipsis"),
        Builder{}
            .inputs({{ET, {2, 2, 1}, std::vector<T>{1, 2, 3, 4}},
                     {ET, {4, 1, 1}, std::vector<T>{1, 2, 3, 4}},
                     {ET,
                      {1, 1, 2, 3, 1, 3},
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}}})
            .equation("a...b,bcd,...dbc->c...a")
            .expectedResult(
                {ET, {3, 1, 1, 2, 2}, std::vector<T>{120, 360, 780, 1560, 150, 450, 840, 1680, 180, 540, 900, 1800}})
            .testcaseName("einsum_3in_broadcast_duplicated_ellipsis"),

        Builder{}
            .inputs({{ET, {2, 2, 1}, std::vector<T>{1, 2, 3, 4}},
                     {ET, {4, 1, 1}, std::vector<T>{1, 2, 3, 4}},
                     {ET, {1, 2, 3, 1, 3, 3}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                             15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                                             29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                                             43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54}}})
            .equation("a...b,bcd,...dbcc->c...a")
            .expectedResult(
                {ET, {3, 1, 2, 2}, std::vector<T>{300, 900, 2220, 4440, 420, 1260, 2460, 4920, 540, 1620, 2700, 5400}})
            .testcaseName("einsum_3in_broadcast_duplicated_ellipsis_repeated_1"),
        Builder{}
            .inputs({{ET, {2, 2, 1, 1, 1}, std::vector<T>{1, 2, 3, 4}},
                     {ET, {4, 1, 1, 1, 1, 1}, std::vector<T>{1, 2, 3, 4}},
                     {ET, {3, 1, 3, 3}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                       15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}}})
            .equation("a...b,bcccdd,...dbcc->cb...")
            .expectedResult(
                {ET, {3, 4, 2, 1, 1}, std::vector<T>{120, 180, 240, 360,  360, 540, 480, 720, 168, 252, 336, 504,
                                                     504, 756, 672, 1008, 216, 324, 432, 648, 648, 972, 864, 1296}})
            .testcaseName("einsum_3in_broadcast_duplicated_ellipsis_repeated_1"),
        Builder{}
            .inputs({{ET, {2, 2, 1}, std::vector<T>{1, 2, 3, 4}},
                     {ET, {4, 1, 1}, std::vector<T>{1, 2, 3, 4}},
                     {ET, {1, 2, 3, 1, 3, 3}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                             15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                                             29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                                             43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54}}})
            .equation("a...b,bcd,...dbcc->ca")
            .expectedResult({ET, {3, 2}, std::vector<T>{2520, 5340, 2880, 6180, 3240, 7020}})
            .testcaseName("einsum_3in_broadcast_duplicated_ellipsis_repeated_3"),
        Builder{}
            .inputs({{ET, {2, 2, 1, 1, 1}, std::vector<T>{1, 2, 3, 4}},
                     {ET, {4, 1, 1, 1, 1, 1}, std::vector<T>{1, 2, 3, 4}},
                     {ET, {3, 1, 3, 3}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                       15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}}})
            .equation("a...b,bcccdd,...dbcc->cb")
            .expectedResult(
                {ET, {3, 4}, std::vector<T>{300, 600, 900, 1200, 420, 840, 1260, 1680, 540, 1080, 1620, 2160}})
            .testcaseName("einsum_3in_broadcast_duplicated_ellipsis_repeated_4"),
        Builder{}
            .inputs({{ET, {2, 1, 4}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8}},
                     {ET, {1, 1, 1, 2, 2}, std::vector<T>{1, 2, 3, 4}}})
            .equation("acd,ad...->acd...")
            .expectedResult(
                {ET, {2, 1, 4, 1, 2, 2}, std::vector<T>{1, 2,  3,  4,  2, 4,  6,  8,  3, 6,  9,  12, 4, 8,  12, 16,
                                                        5, 10, 15, 20, 6, 12, 18, 24, 7, 14, 21, 28, 8, 16, 24, 32}})
            .testcaseName("einsum_no_reduce1_sep2ellipsis"),
        Builder{}
            .inputs({{ET, {1, 5}, std::vector<T>{1, 2, 3, 4, 5}},
                     {ET, {1, 1, 2, 1}, std::vector<T>{1, 2}},
                     {ET, {2, 2, 1, 4}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
                     {ET, {}, std::vector<T>{2}},
                     {ET, {2, 1, 1}, std::vector<T>{1, 2}}})
            .equation("a...b,b...,aacd,,...dd->da...c")
            .expectedResult(
                {ET, {4, 2, 1, 2, 2, 1}, std::vector<T>{30,  60,   60,  120, 390,  780, 780, 1560, 60,  120, 120,
                                                        240, 420,  840, 840, 1680, 90,  180, 180,  360, 450, 900,
                                                        900, 1800, 120, 240, 240,  480, 480, 960,  960, 1920}})
            .testcaseName("einsum_multi_input_broadcasting")

    };
    return params;
}

std::vector<EinsumParams> generateCombinedParams() {
    const std::vector<std::vector<EinsumParams>> generatedParams{
        generateParams<element::Type_t::i32>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
    };
    std::vector<EinsumParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Einsum_With_Hardcoded_Refs,
                         ReferenceEinsumTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceEinsumTest::getTestCaseName);
}  // namespace
