// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct UniqueParams {
    template <typename Data_t, typename Index_t>
    UniqueParams(const Shape& data_shape,
                 const std::vector<Data_t>& input_data,
                 const std::vector<Data_t>& expected_unique_values,
                 const std::vector<Index_t>& expected_indices,
                 const std::vector<Index_t>& expected_rev_indices,
                 const std::vector<int64_t>& expected_counts,
                 const bool sorted = true,
                 const std::string& tested_case = "")
        : m_data_shape{data_shape},
          m_data_type{element::from<Data_t>()},
          m_index_type{element::from<Index_t>()},
          m_input_data{CreateTensor(m_data_type, input_data)},
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
        if (!param.m_tested_case.empty()) {
            result << "tested_case=" << param.m_tested_case << "; ";
        }

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const UniqueParams& params) {
        const auto in = std::make_shared<op::v0::Parameter>(params.m_data_type, params.m_data_shape);
        const auto unique = std::make_shared<op::v10::Unique>(in, params.m_sorted, params.m_index_type);
        return std::make_shared<ov::Model>(unique, ParameterVector{in});
    }
};

TEST_P(ReferenceUniqueLayerTest_NoAxis, CompareWithHardcodedRefs) {
    Exec();
}

template <typename Data_t, typename Index_t>
std::vector<UniqueParams> params_unique_int() {
    static_assert(std::numeric_limits<Data_t>::is_integer);
    const std::vector<UniqueParams> params{UniqueParams{Shape{},
                                                        std::vector<Data_t>{1},
                                                        std::vector<Data_t>{1},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1},
                                                        false},
                                           UniqueParams{Shape{},
                                                        std::vector<Data_t>{1},
                                                        std::vector<Data_t>{1},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1},
                                                        true},
                                           UniqueParams{Shape{1},
                                                        std::vector<Data_t>{2},
                                                        std::vector<Data_t>{2},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1},
                                                        false},
                                           UniqueParams{Shape{1},
                                                        std::vector<Data_t>{2},
                                                        std::vector<Data_t>{2},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1},
                                                        true},
                                           UniqueParams{Shape{5},
                                                        std::vector<Data_t>{5, 4, 3, 2, 1},
                                                        std::vector<Data_t>{5, 4, 3, 2, 1},
                                                        std::vector<Index_t>{0, 1, 2, 3, 4},
                                                        std::vector<Index_t>{0, 1, 2, 3, 4},
                                                        std::vector<int64_t>{1, 1, 1, 1, 1},
                                                        false,
                                                        "1D no duplicates"},
                                           UniqueParams{Shape{5},
                                                        std::vector<Data_t>{5, 4, 3, 2, 1},
                                                        std::vector<Data_t>{1, 2, 3, 4, 5},
                                                        std::vector<Index_t>{4, 3, 2, 1, 0},
                                                        std::vector<Index_t>{4, 3, 2, 1, 0},
                                                        std::vector<int64_t>{1, 1, 1, 1, 1},
                                                        true,
                                                        "1D no duplicates"},
                                           UniqueParams{Shape{7},
                                                        std::vector<Data_t>{1, 3, 5, 3, 2, 4, 2},
                                                        std::vector<Data_t>{1, 3, 5, 2, 4},
                                                        std::vector<Index_t>{0, 1, 2, 4, 5},
                                                        std::vector<Index_t>{0, 1, 2, 1, 3, 4, 3},
                                                        std::vector<int64_t>{1, 2, 1, 2, 1},
                                                        false,
                                                        "1D with duplicates"},
                                           UniqueParams{Shape{7},
                                                        std::vector<Data_t>{1, 3, 5, 3, 2, 4, 2},
                                                        std::vector<Data_t>{1, 2, 3, 4, 5},
                                                        std::vector<Index_t>{0, 4, 1, 5, 2},
                                                        std::vector<Index_t>{0, 2, 4, 2, 1, 3, 1},
                                                        std::vector<int64_t>{1, 2, 2, 1, 1},
                                                        true,
                                                        "1D with duplicates"}};

    return params;
}

template <typename Data_t, typename Index_t>
std::vector<UniqueParams> params_unique_float() {
    static_assert(!std::numeric_limits<Data_t>::is_integer);
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
                                                        false},
                                           UniqueParams{Shape{},
                                                        std::vector<Data_t>{pi},
                                                        std::vector<Data_t>{pi},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1},
                                                        true},
                                           UniqueParams{Shape{1},
                                                        std::vector<Data_t>{-e},
                                                        std::vector<Data_t>{-e},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1},
                                                        false},
                                           UniqueParams{Shape{1},
                                                        std::vector<Data_t>{-e},
                                                        std::vector<Data_t>{-e},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1},
                                                        true},
                                           UniqueParams{Shape{6},
                                                        std::vector<Data_t>{pi, -pi, -e, e, sq3, sq2},
                                                        std::vector<Data_t>{pi, -pi, -e, e, sq3, sq2},
                                                        std::vector<Index_t>{0, 1, 2, 3, 4, 5},
                                                        std::vector<Index_t>{0, 1, 2, 3, 4, 5},
                                                        std::vector<int64_t>{1, 1, 1, 1, 1, 1},
                                                        false,
                                                        "1D no duplicates"},
                                           UniqueParams{Shape{6},
                                                        std::vector<Data_t>{pi, -pi, -e, e, sq3, sq2},
                                                        std::vector<Data_t>{-pi, -e, sq2, sq3, e, pi},
                                                        std::vector<Index_t>{1, 2, 5, 4, 3, 0},
                                                        std::vector<Index_t>{5, 0, 1, 4, 3, 2},
                                                        std::vector<int64_t>{1, 1, 1, 1, 1, 1},
                                                        true,
                                                        "1D no duplicates"}};

    return params;
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

INSTANTIATE_TEST_SUITE_P(smoke_ReferenceUniqueLayerTest_NoAxis,
                         ReferenceUniqueLayerTest_NoAxis,
                         ::testing::ValuesIn(flatten({params_unique_float<float, int32_t>(),
                                                      params_unique_float<float, int64_t>(),
                                                      params_unique_int<int32_t, int32_t>(),
                                                      params_unique_int<int32_t, int64_t>()})),
                         ReferenceUniqueLayerTest_NoAxis::getTestCaseName);

}  // namespace
