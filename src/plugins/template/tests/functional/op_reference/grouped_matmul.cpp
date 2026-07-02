// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grouped_matmul.hpp"

namespace reference_tests {
namespace GroupedMatMulRefTestDefinitions {
namespace {

TEST_P(ReferenceGroupedMatMulTest, CompareWithRefs) {
    Exec();
}

template <ov::element::Type_t ET>
std::vector<GroupedMatMulParams> generateParams() {
    using T = typename ov::element_type_traits<ET>::value_type;

    std::vector<GroupedMatMulParams> params;
    const std::string type_suffix = "_" + ov::element::Type(ET).get_type_name();

    // Case: 3D × 3D batched - simple 2 groups, 2×2 matmul each
    // Group 0: [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
    // Group 1: [[5,6],[7,8]] @ [[5,6],[7,8]] = [[67,78],[91,106]]
    // mat_b stored as [G, N, K]: group 0 = [[1,3],[2,4]], group 1 = [[5,7],[6,8]]
    params.push_back(GroupedMatMulParams(ov::Shape{2, 2, 2},  // mat_a: (G=2, M=2, K=2)
                                         ov::Shape{2, 2, 2},  // mat_b: (G=2, N=2, K=2)
                                         ov::Shape{2, 2, 2},  // output: (G=2, M=2, N=2)
                                         ET,
                                         std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},  // mat_a
                                         std::vector<T>{1, 3, 2, 4, 5, 7, 6, 8},  // mat_b (transposed per group)
                                         std::vector<T>{7, 10, 15, 22, 67, 78, 91, 106},  // expected
                                         "3D_3D_2groups_2x2" + type_suffix));

    // Case: 3D × 3D - single group
    // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
    // mat_b stored as [G, N, K]: group 0 = [[1,3,5],[2,4,6]]
    params.push_back(GroupedMatMulParams(ov::Shape{1, 2, 3},  // mat_a: (G=1, M=2, K=3)
                                         ov::Shape{1, 2, 3},  // mat_b: (G=1, N=2, K=3)
                                         ov::Shape{1, 2, 2},  // output: (G=1, M=2, N=2)
                                         ET,
                                         std::vector<T>{1, 2, 3, 4, 5, 6},  // mat_a
                                         std::vector<T>{1, 3, 5, 2, 4, 6},  // mat_b (transposed)
                                         std::vector<T>{22, 28, 49, 64},    // expected
                                         "3D_3D_1group_2x3_3x2" + type_suffix));

    // Case: 2D × 3D with offsets - 2 experts, [2,1] tokens per expert
    // Expert 0 gets rows [0:2], Expert 1 gets rows [2:3]
    // mat_a[:2] @ mat_b[0] = [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
    // mat_a[2:3] @ mat_b[1] = [[5,6]] @ [[5,6],[7,8]] = [[67,78]]
    // mat_b stored as [G, N, K]: expert 0 = [[1,3],[2,4]], expert 1 = [[5,7],[6,8]]
    params.push_back(GroupedMatMulParams(ov::Shape{3, 2},     // mat_a: (total_tokens=3, K=2)
                                         ov::Shape{2, 2, 2},  // mat_b: (G=2, N=2, K=2)
                                         ov::Shape{2},        // offsets: cumulative [2, 3]
                                         ov::Shape{3, 2},     // output: (total_tokens=3, N=2)
                                         ET,
                                         ov::element::i32,
                                         std::vector<T>{1, 2, 3, 4, 5, 6},        // mat_a
                                         std::vector<T>{1, 3, 2, 4, 5, 7, 6, 8},  // mat_b (transposed per group)
                                         std::vector<int32_t>{2, 3},              // offsets
                                         std::vector<T>{7, 10, 15, 22, 67, 78},   // expected
                                         "2D_3D_2experts_offsets" + type_suffix));

    // Case: 2D × 3D - 3 experts with varying tokens [1, 2, 1]
    params.push_back(GroupedMatMulParams(
        ov::Shape{4, 2},     // mat_a: (total_tokens=4, K=2)
        ov::Shape{3, 2, 2},  // mat_b: (G=3, K=2, N=2)
        ov::Shape{3},        // offsets
        ov::Shape{4, 2},     // output
        ET,
        ov::element::i64,
        std::vector<T>{1, 0, 0, 1, 1, 1, 2, 2},              // mat_a: 4 tokens
        std::vector<T>{1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3},  // mat_b: 3 experts, each 2x2 identity-like
        std::vector<int64_t>{1, 3, 4},                       // offsets: [1, 3, 4]
        std::vector<T>{1, 0, 0, 2, 2, 2, 6, 6},              // expected
        "2D_3D_3experts_varying_tokens" + type_suffix));

    return params;
}

std::vector<GroupedMatMulParams> generateCombinedParams() {
    std::vector<std::vector<GroupedMatMulParams>> params_list;
    params_list.push_back(generateParams<ov::element::Type_t::f32>());
    params_list.push_back(generateParams<ov::element::Type_t::i32>());

    std::vector<GroupedMatMulParams> combined;
    for (auto& params : params_list) {
        combined.insert(combined.end(), params.begin(), params.end());
    }
    return combined;
}

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_With_Hardcoded_Refs,
                         ReferenceGroupedMatMulTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceGroupedMatMulTest::getTestCaseName);

}  // namespace
}  // namespace GroupedMatMulRefTestDefinitions
}  // namespace reference_tests
