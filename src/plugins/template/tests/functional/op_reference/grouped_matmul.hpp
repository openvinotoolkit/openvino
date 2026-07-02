// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/parameter.hpp"

namespace reference_tests {
namespace GroupedMatMulRefTestDefinitions {

struct GroupedMatMulParams {
    // Case: 2D × 3D with offsets
    template <class T, class TIdx>
    GroupedMatMulParams(const ov::Shape& mat_a_shape,
                        const ov::Shape& mat_b_shape,
                        const ov::Shape& offsets_shape,
                        const ov::Shape& expected_shape,
                        const ov::element::Type_t& data_type,
                        const ov::element::Type_t& offsets_type,
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

    // Case: 3D × 3D without offsets
    template <class T>
    GroupedMatMulParams(const ov::Shape& mat_a_shape,
                        const ov::Shape& mat_b_shape,
                        const ov::Shape& expected_shape,
                        const ov::element::Type_t& data_type,
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

    ov::Shape m_mat_a_shape;
    ov::Shape m_mat_b_shape;
    ov::Shape m_offsets_shape;
    ov::Shape m_expected_shape;
    ov::element::Type_t m_data_type;
    ov::element::Type_t m_offsets_type;
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
    static std::shared_ptr<ov::Model> CreateFunction2D3D(const GroupedMatMulParams& p) {
        auto mat_a = std::make_shared<ov::op::v0::Parameter>(p.m_data_type, p.m_mat_a_shape);
        auto mat_b = std::make_shared<ov::op::v0::Parameter>(p.m_data_type, p.m_mat_b_shape);
        auto offsets = std::make_shared<ov::op::v0::Parameter>(p.m_offsets_type, p.m_offsets_shape);
        auto grouped_matmul = std::make_shared<ov::op::v17::GroupedMatMul>(mat_a, mat_b, offsets);
        return std::make_shared<ov::Model>(grouped_matmul, ov::ParameterVector{mat_a, mat_b, offsets});
    }

    static std::shared_ptr<ov::Model> CreateFunction3D3D(const GroupedMatMulParams& p) {
        auto mat_a = std::make_shared<ov::op::v0::Parameter>(p.m_data_type, p.m_mat_a_shape);
        auto mat_b = std::make_shared<ov::op::v0::Parameter>(p.m_data_type, p.m_mat_b_shape);
        auto grouped_matmul = std::make_shared<ov::op::v17::GroupedMatMul>(mat_a, mat_b);
        return std::make_shared<ov::Model>(grouped_matmul, ov::ParameterVector{mat_a, mat_b});
    }
};

}  // namespace GroupedMatMulRefTestDefinitions
}  // namespace reference_tests
