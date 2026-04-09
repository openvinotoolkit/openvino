// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grouped_matmul.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct GroupedMatMulParams {
    // Case 1: 2D × 3D with offsets
    template <class T, class TIdx>
    GroupedMatMulParams(const Shape& mat_a_shape,
                        const Shape& mat_b_shape,
                        const Shape& offsets_shape,
                        const Shape& expected_shape,
                        const element::Type_t& data_type,
                        const element::Type_t& offsets_type,
                        const std::vector<T>& mat_a_values,
                        const std::vector<T>& mat_b_values,
                        const std::vector<TIdx>& offsets_values,
                        const std::vector<T>& expected_values,
                        const std::string& test_name)
        : m_mat_a_shape(mat_a_shape),
          m_mat_b_shape(mat_b_shape),
          m_offsets_shape(offsets_shape),
          m_expected_shape(expected_shape),
          m_data_type(data_type),
          m_offsets_type(offsets_type),
          m_has_offsets(true),
          m_test_name(test_name) {
        m_mat_a_tensor = CreateTensor(mat_a_shape, data_type, mat_a_values);
        m_mat_b_tensor = CreateTensor(mat_b_shape, data_type, mat_b_values);
        m_offsets_tensor = CreateTensor(offsets_shape, offsets_type, offsets_values);
        m_expected_tensor = CreateTensor(expected_shape, data_type, expected_values);
    }

    // Case 2: 3D × 3D without offsets
    template <class T>
    GroupedMatMulParams(const Shape& mat_a_shape,
                        const Shape& mat_b_shape,
                        const Shape& expected_shape,
                        const element::Type_t& data_type,
                        const std::vector<T>& mat_a_values,
                        const std::vector<T>& mat_b_values,
                        const std::vector<T>& expected_values,
                        const std::string& test_name)
        : m_mat_a_shape(mat_a_shape),
          m_mat_b_shape(mat_b_shape),
          m_expected_shape(expected_shape),
          m_data_type(data_type),
          m_has_offsets(false),
          m_test_name(test_name) {
        m_mat_a_tensor = CreateTensor(mat_a_shape, data_type, mat_a_values);
        m_mat_b_tensor = CreateTensor(mat_b_shape, data_type, mat_b_values);
        m_expected_tensor = CreateTensor(expected_shape, data_type, expected_values);
    }

    Shape m_mat_a_shape;
    Shape m_mat_b_shape;
    Shape m_offsets_shape;
    Shape m_expected_shape;
    element::Type_t m_data_type;
    element::Type_t m_offsets_type;
    bool m_has_offsets;
    ov::Tensor m_mat_a_tensor;
    ov::Tensor m_mat_b_tensor;
    ov::Tensor m_offsets_tensor;
    ov::Tensor m_expected_tensor;
    std::string m_test_name;
};

class ReferenceGroupedMatMulTest : public testing::TestWithParam<GroupedMatMulParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        if (params.m_has_offsets) {
            function = CreateFunction2D3D(params);
            inputData = {params.m_mat_a_tensor, params.m_mat_b_tensor, params.m_offsets_tensor};
        } else {
            function = CreateFunction3D3D(params);
            inputData = {params.m_mat_a_tensor, params.m_mat_b_tensor};
        }
        refOutData = {params.m_expected_tensor};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GroupedMatMulParams>& obj) {
        return obj.param.m_test_name;
    }

private:
    static std::shared_ptr<Model> CreateFunction2D3D(const GroupedMatMulParams& p) {
        auto mat_a = std::make_shared<op::v0::Parameter>(p.m_data_type, p.m_mat_a_shape);
        auto mat_b = std::make_shared<op::v0::Parameter>(p.m_data_type, p.m_mat_b_shape);
        auto offsets = std::make_shared<op::v0::Parameter>(p.m_offsets_type, p.m_offsets_shape);
        auto grouped_matmul = std::make_shared<op::v17::GroupedMatMul>(mat_a, mat_b, offsets);
        return std::make_shared<Model>(grouped_matmul, ParameterVector{mat_a, mat_b, offsets});
    }

    static std::shared_ptr<Model> CreateFunction3D3D(const GroupedMatMulParams& p) {
        auto mat_a = std::make_shared<op::v0::Parameter>(p.m_data_type, p.m_mat_a_shape);
        auto mat_b = std::make_shared<op::v0::Parameter>(p.m_data_type, p.m_mat_b_shape);
        auto grouped_matmul = std::make_shared<op::v17::GroupedMatMul>(mat_a, mat_b);
        return std::make_shared<Model>(grouped_matmul, ParameterVector{mat_a, mat_b});
    }
};

TEST_P(ReferenceGroupedMatMulTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<GroupedMatMulParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<GroupedMatMulParams> params;
    const std::string type_suffix = "_" + element::Type(ET).get_type_name();

    // Case 2: 3D × 3D batched - simple 2 groups, 2×2 matmul each
    // Group 0: [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
    // Group 1: [[5,6],[7,8]] @ [[5,6],[7,8]] = [[67,78],[91,106]]
    params.push_back(GroupedMatMulParams(
        Shape{2, 2, 2},                                                    // mat_a: (G=2, M=2, K=2)
        Shape{2, 2, 2},                                                    // mat_b: (G=2, K=2, N=2)
        Shape{2, 2, 2},                                                    // output: (G=2, M=2, N=2)
        ET,
        std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},                            // mat_a
        std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},                            // mat_b
        std::vector<T>{7, 10, 15, 22, 67, 78, 91, 106},                    // expected
        "3D_3D_2groups_2x2" + type_suffix));

    // Case 2: 3D × 3D - single group
    // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
    params.push_back(GroupedMatMulParams(
        Shape{1, 2, 3},                                                    // mat_a: (G=1, M=2, K=3)
        Shape{1, 3, 2},                                                    // mat_b: (G=1, K=3, N=2)
        Shape{1, 2, 2},                                                    // output: (G=1, M=2, N=2)
        ET,
        std::vector<T>{1, 2, 3, 4, 5, 6},                                  // mat_a
        std::vector<T>{1, 2, 3, 4, 5, 6},                                  // mat_b
        std::vector<T>{22, 28, 49, 64},                                    // expected
        "3D_3D_1group_2x3_3x2" + type_suffix));

    // Case 1: 2D × 3D with offsets - 2 experts, [2,1] tokens per expert
    // Expert 0 gets rows [0:2], Expert 1 gets rows [2:3]
    // mat_a[:2] @ mat_b[0] = [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
    // mat_a[2:3] @ mat_b[1] = [[5,6]] @ [[5,6],[7,8]] = [[67,78]]
    params.push_back(GroupedMatMulParams(
        Shape{3, 2},                                                       // mat_a: (total_tokens=3, K=2)
        Shape{2, 2, 2},                                                    // mat_b: (G=2, K=2, N=2)
        Shape{2},                                                          // offsets: cumulative [2, 3]
        Shape{3, 2},                                                       // output: (total_tokens=3, N=2)
        ET,
        element::i32,
        std::vector<T>{1, 2, 3, 4, 5, 6},                                  // mat_a
        std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},                            // mat_b
        std::vector<int32_t>{2, 3},                                        // offsets
        std::vector<T>{7, 10, 15, 22, 67, 78},                             // expected
        "2D_3D_2experts_offsets" + type_suffix));

    // Case 1: 2D × 3D - 3 experts with varying tokens [1, 2, 1]
    params.push_back(GroupedMatMulParams(
        Shape{4, 2},                                                       // mat_a: (total_tokens=4, K=2)
        Shape{3, 2, 2},                                                    // mat_b: (G=3, K=2, N=2)
        Shape{3},                                                          // offsets
        Shape{4, 2},                                                       // output
        ET,
        element::i64,
        std::vector<T>{1, 0, 0, 1, 1, 1, 2, 2},                            // mat_a: 4 tokens
        std::vector<T>{1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3},                // mat_b: 3 experts, each 2x2 identity-like
        std::vector<int64_t>{1, 3, 4},                                     // offsets: [1, 3, 4]
        std::vector<T>{1, 0, 0, 2, 2, 2, 6, 6},                            // expected
        "2D_3D_3experts_varying_tokens" + type_suffix));

    return params;
}

std::vector<GroupedMatMulParams> generateCombinedParams() {
    std::vector<std::vector<GroupedMatMulParams>> params_list;
    params_list.push_back(generateParams<element::Type_t::f32>());
    params_list.push_back(generateParams<element::Type_t::i32>());

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
