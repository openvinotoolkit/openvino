// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

std::shared_ptr<op::v0::Constant> make_axis(const int64_t axis, const element::Type& et = element::i32) {
    return op::v0::Constant::create(et, Shape{}, {axis});
}

struct UniqueParams {
    template <typename Data_t, typename Index_t>
    UniqueParams(const Shape& data_shape,
                 const std::vector<Data_t>& input_data,
                 const std::vector<Data_t>& expected_unique_values,
                 const std::vector<Index_t>& expected_indices,
                 const std::vector<Index_t>& expected_rev_indices,
                 const std::vector<int64_t>& expected_counts,
                 std::shared_ptr<op::v0::Constant> axis_descritptor = nullptr,
                 const bool sorted = true,
                 const std::string& tested_case = "")
        : m_data_shape{data_shape},
          m_data_type{element::from<Data_t>()},
          m_index_type{element::from<Index_t>()},
          m_input_data{CreateTensor(m_data_type, input_data)},
          m_axis{axis_descritptor},
          m_sorted{sorted},
          m_tested_case{tested_case} {
        m_expected_outputs[0] = CreateTensor(m_data_type, expected_unique_values);
        m_expected_outputs[1] = CreateTensor(m_index_type, expected_indices);
        m_expected_outputs[2] = CreateTensor(m_index_type, expected_rev_indices);
        m_expected_outputs[3] = CreateTensor(element::i64, expected_counts);
    }

    Shape m_data_shape;
    element::Type m_data_type;
    element::Type m_index_type;
    ov::Tensor m_input_data;
    ov::TensorVector m_expected_outputs = ov::TensorVector(4);
    std::shared_ptr<op::v0::Constant> m_axis = nullptr;
    bool m_sorted;
    std::string m_tested_case;
};

class ReferenceUniqueLayerTest_NoAxis : public testing::TestWithParam<UniqueParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.m_input_data};
        refOutData = params.m_expected_outputs;
    }

    static std::string getTestCaseName(const testing::TestParamInfo<UniqueParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;

        result << "data_shape=" << param.m_data_shape << "; ";
        result << "data_type=" << param.m_data_type << "; ";
        result << "index_type=" << param.m_index_type << "; ";
        result << "sorted=" << param.m_sorted << "; ";
        if (param.m_axis) {
            result << "axis=" << param.m_axis->cast_vector<int64_t>()[0] << "; ";
        }
        if (!param.m_tested_case.empty()) {
            result << "tested_case=" << param.m_tested_case << "; ";
        }

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const UniqueParams& params) {
        const auto in = std::make_shared<op::v0::Parameter>(params.m_data_type, params.m_data_shape);
        std::shared_ptr<Node> unique;
        if (params.m_axis) {
            unique = std::make_shared<op::v10::Unique>(in, params.m_axis, params.m_sorted, params.m_index_type);
        } else {
            unique = std::make_shared<op::v10::Unique>(in, params.m_sorted, params.m_index_type);
        }
        return std::make_shared<ov::Model>(unique, ParameterVector{in});
    }
};

TEST_P(ReferenceUniqueLayerTest_NoAxis, CompareWithHardcodedRefs) {
    Exec();
}

template <typename T>
std::vector<T> flatten(std::initializer_list<std::vector<T>> test_cases) {
    using std::begin;
    using std::end;

    std::vector<T> flattened;
    for (auto&& tc : test_cases) {
        flattened.insert(flattened.end(), std::make_move_iterator(begin(tc)), std::make_move_iterator(end(tc)));
    }
    return flattened;
}

template <typename Data_t, typename Index_t>
std::vector<UniqueParams> params_unique_int() {
    static_assert(std::numeric_limits<Data_t>::is_integer, "Integer type expected");
    std::vector<UniqueParams> scalar_and_1D{UniqueParams{Shape{},
                                                         std::vector<Data_t>{1},
                                                         std::vector<Data_t>{1},
                                                         std::vector<Index_t>{0},
                                                         std::vector<Index_t>{0},
                                                         std::vector<int64_t>{1},
                                                         nullptr,
                                                         false},
                                            UniqueParams{Shape{},
                                                         std::vector<Data_t>{1},
                                                         std::vector<Data_t>{1},
                                                         std::vector<Index_t>{0},
                                                         std::vector<Index_t>{0},
                                                         std::vector<int64_t>{1},
                                                         nullptr,
                                                         true},
                                            UniqueParams{Shape{1},
                                                         std::vector<Data_t>{2},
                                                         std::vector<Data_t>{2},
                                                         std::vector<Index_t>{0},
                                                         std::vector<Index_t>{0},
                                                         std::vector<int64_t>{1},
                                                         nullptr,
                                                         false},
                                            UniqueParams{Shape{1},
                                                         std::vector<Data_t>{2},
                                                         std::vector<Data_t>{2},
                                                         std::vector<Index_t>{0},
                                                         std::vector<Index_t>{0},
                                                         std::vector<int64_t>{1},
                                                         nullptr,
                                                         true},
                                            UniqueParams{Shape{5},
                                                         std::vector<Data_t>{5, 4, 3, 2, 1},
                                                         std::vector<Data_t>{5, 4, 3, 2, 1},
                                                         std::vector<Index_t>{0, 1, 2, 3, 4},
                                                         std::vector<Index_t>{0, 1, 2, 3, 4},
                                                         std::vector<int64_t>{1, 1, 1, 1, 1},
                                                         nullptr,
                                                         false,
                                                         "1D no duplicates"},
                                            UniqueParams{Shape{5},
                                                         std::vector<Data_t>{5, 4, 3, 2, 1},
                                                         std::vector<Data_t>{1, 2, 3, 4, 5},
                                                         std::vector<Index_t>{4, 3, 2, 1, 0},
                                                         std::vector<Index_t>{4, 3, 2, 1, 0},
                                                         std::vector<int64_t>{1, 1, 1, 1, 1},
                                                         nullptr,
                                                         true,
                                                         "1D no duplicates"},
                                            UniqueParams{Shape{7},
                                                         std::vector<Data_t>{1, 3, 5, 3, 2, 4, 2},
                                                         std::vector<Data_t>{1, 3, 5, 2, 4},
                                                         std::vector<Index_t>{0, 1, 2, 4, 5},
                                                         std::vector<Index_t>{0, 1, 2, 1, 3, 4, 3},
                                                         std::vector<int64_t>{1, 2, 1, 2, 1},
                                                         nullptr,
                                                         false,
                                                         "1D with duplicates"},
                                            UniqueParams{Shape{7},
                                                         std::vector<Data_t>{1, 3, 5, 3, 2, 4, 2},
                                                         std::vector<Data_t>{1, 2, 3, 4, 5},
                                                         std::vector<Index_t>{0, 4, 1, 5, 2},
                                                         std::vector<Index_t>{0, 2, 4, 2, 1, 3, 1},
                                                         std::vector<int64_t>{1, 2, 2, 1, 1},
                                                         nullptr,
                                                         true,
                                                         "1D with duplicates"},
                                            UniqueParams{Shape{7},
                                                         std::vector<Data_t>{1, 3, 5, 3, 2, 4, 2},
                                                         std::vector<Data_t>{1, 2, 3, 4, 5},
                                                         std::vector<Index_t>{0, 4, 1, 5, 2},
                                                         std::vector<Index_t>{0, 2, 4, 2, 1, 3, 1},
                                                         std::vector<int64_t>{1, 2, 2, 1, 1},
                                                         make_axis(0),
                                                         true,
                                                         "1D with duplicates and axis"}};

    std::vector<UniqueParams> N_C_layout{UniqueParams{Shape{2, 6},
                                                      std::vector<Data_t>{3, 5, 3, 2, 4, 2, 1, 2, 3, 4, 5, 6},
                                                      std::vector<Data_t>{3, 5, 2, 4, 1, 6},
                                                      std::vector<Index_t>{0, 1, 3, 4, 6, 11},
                                                      std::vector<Index_t>{0, 1, 0, 2, 3, 2, 4, 2, 0, 3, 1, 5},
                                                      std::vector<int64_t>{3, 2, 3, 2, 1, 1},
                                                      nullptr,
                                                      false,
                                                      "2D no axis"},
                                         UniqueParams{Shape{2, 4},
                                                      std::vector<Data_t>{1, 2, 3, 4, 1, 2, 3, 5},
                                                      std::vector<Data_t>{1, 2, 3, 4, 1, 2, 3, 5},
                                                      std::vector<Index_t>{0, 1},
                                                      std::vector<Index_t>{0, 1},
                                                      std::vector<int64_t>{1, 1},
                                                      make_axis(0),
                                                      false,
                                                      "2D no duplicates"},
                                         UniqueParams{Shape{2, 4},
                                                      std::vector<Data_t>{1, 2, 3, 4, 1, 2, 3, 5},
                                                      std::vector<Data_t>{1, 2, 3, 4, 1, 2, 3, 5},
                                                      std::vector<Index_t>{0, 1, 2, 3},
                                                      std::vector<Index_t>{0, 1, 2, 3},
                                                      std::vector<int64_t>{1, 1, 1, 1},
                                                      make_axis(1),
                                                      false,
                                                      "2D no duplicates"},
                                         UniqueParams{Shape{2, 4},
                                                      std::vector<Data_t>{1, 2, 2, 4, 1, 2, 2, 5},
                                                      std::vector<Data_t>{1, 2, 4, 1, 2, 5},
                                                      std::vector<Index_t>{0, 1, 3},
                                                      std::vector<Index_t>{0, 1, 1, 2},
                                                      std::vector<int64_t>{1, 2, 1},
                                                      make_axis(1),
                                                      false,
                                                      "2D with duplicates"}};

    std::vector<UniqueParams> N_D_layout{UniqueParams{Shape{2, 2, 3},
                                                      // 2 identical 2D slices over axis 0
                                                      std::vector<Data_t>{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6},
                                                      std::vector<Data_t>{1, 2, 3, 4, 5, 6},
                                                      std::vector<Index_t>{0},
                                                      std::vector<Index_t>{0, 0},
                                                      std::vector<int64_t>{2},
                                                      make_axis(0),
                                                      false,
                                                      "3D with duplicates"},
                                         UniqueParams{Shape{2, 2, 3},
                                                      // 2 identical 2D slices over axis 1
                                                      std::vector<Data_t>{6, 5, 4, 6, 5, 4, 3, 2, 1, 3, 2, 1},
                                                      std::vector<Data_t>{6, 5, 4, 3, 2, 1},
                                                      std::vector<Index_t>{0},
                                                      std::vector<Index_t>{0, 0},
                                                      std::vector<int64_t>{2},
                                                      make_axis(1),
                                                      false,
                                                      "3D with duplicates"},
                                         UniqueParams{Shape{2, 2, 3},
                                                      // the first and the last slice over axis 2 are equal
                                                      std::vector<Data_t>{-1, 2, -1, 5, -3, 5, 7, -8, 7, 4, 4, 4},
                                                      std::vector<Data_t>{-1, 2, 5, -3, 7, -8, 4, 4},
                                                      std::vector<Index_t>{0, 1},
                                                      std::vector<Index_t>{0, 1, 0},
                                                      std::vector<int64_t>{2, 1},
                                                      make_axis(2),
                                                      false,
                                                      "3D with duplicates(1 & 3)"},
                                         UniqueParams{Shape{2, 2, 3},
                                                      // the first and the second slice over axis 2 are equal
                                                      std::vector<Data_t>{-1, -1, 2, 5, 5, -3, 7, 7, -8, 4, 4, 4},
                                                      std::vector<Data_t>{-1, 2, 5, -3, 7, -8, 4, 4},
                                                      std::vector<Index_t>{0, 2},
                                                      std::vector<Index_t>{0, 0, 1},
                                                      std::vector<int64_t>{2, 1},
                                                      make_axis(2),
                                                      false,
                                                      "3D with duplicates (1 & 2)"}};

    return flatten({std::move(scalar_and_1D), std::move(N_C_layout), std::move(N_D_layout)});
}

template <typename Data_t, typename Index_t>
std::vector<UniqueParams> params_unique_float() {
    static_assert(!std::numeric_limits<Data_t>::is_integer, "Floating point type expected");
    // just some fancy numbers to be used in the input tensors
    const auto sq2 = Data_t{1.4142135};
    const auto sq3 = Data_t{1.7320508075};
    const auto e = Data_t{2.71828};
    const auto pi = Data_t{3.141592};

    const std::vector<UniqueParams> params{UniqueParams{Shape{},
                                                        std::vector<Data_t>{pi},
                                                        std::vector<Data_t>{pi},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1},
                                                        nullptr,
                                                        false},
                                           UniqueParams{Shape{},
                                                        std::vector<Data_t>{pi},
                                                        std::vector<Data_t>{pi},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1},
                                                        nullptr,
                                                        true},
                                           UniqueParams{Shape{1},
                                                        std::vector<Data_t>{-e},
                                                        std::vector<Data_t>{-e},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1},
                                                        nullptr,
                                                        false},
                                           UniqueParams{Shape{1},
                                                        std::vector<Data_t>{-e},
                                                        std::vector<Data_t>{-e},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1},
                                                        nullptr,
                                                        true},
                                           UniqueParams{Shape{6},
                                                        std::vector<Data_t>{pi, -pi, -e, e, sq3, sq2},
                                                        std::vector<Data_t>{pi, -pi, -e, e, sq3, sq2},
                                                        std::vector<Index_t>{0, 1, 2, 3, 4, 5},
                                                        std::vector<Index_t>{0, 1, 2, 3, 4, 5},
                                                        std::vector<int64_t>{1, 1, 1, 1, 1, 1},
                                                        nullptr,
                                                        false,
                                                        "1D no duplicates"},
                                           UniqueParams{Shape{6},
                                                        std::vector<Data_t>{pi, -pi, -e, e, sq3, sq2},
                                                        std::vector<Data_t>{-pi, -e, sq2, sq3, e, pi},
                                                        std::vector<Index_t>{1, 2, 5, 4, 3, 0},
                                                        std::vector<Index_t>{5, 0, 1, 4, 3, 2},
                                                        std::vector<int64_t>{1, 1, 1, 1, 1, 1},
                                                        nullptr,
                                                        true,
                                                        "1D no duplicates"}};

    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_ReferenceUniqueLayerTest_NoAxis,
                         ReferenceUniqueLayerTest_NoAxis,
                         ::testing::ValuesIn(flatten({params_unique_float<float16, int32_t>(),
                                                      params_unique_float<float16, int64_t>(),
                                                      params_unique_float<bfloat16, int32_t>(),
                                                      params_unique_float<bfloat16, int64_t>(),
                                                      params_unique_float<float, int32_t>(),
                                                      params_unique_float<float, int64_t>(),
                                                      params_unique_float<double, int32_t>(),
                                                      params_unique_float<double, int64_t>(),
                                                      params_unique_int<int16_t, int32_t>(),
                                                      params_unique_int<int8_t, int64_t>(),
                                                      params_unique_int<int8_t, int32_t>(),
                                                      params_unique_int<int16_t, int64_t>(),
                                                      params_unique_int<int32_t, int32_t>(),
                                                      params_unique_int<int32_t, int64_t>(),
                                                      params_unique_int<int64_t, int32_t>(),
                                                      params_unique_int<int64_t, int64_t>()})),

                         ReferenceUniqueLayerTest_NoAxis::getTestCaseName);

}  // namespace
