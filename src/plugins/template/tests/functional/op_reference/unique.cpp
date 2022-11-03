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
                 const std::vector<int64_t>& expected_counts)
        : m_data_shape(data_shape),
          m_data_type(element::from<Data_t>()),
          m_index_type(element::from<Index_t>()),
          m_input_data(CreateTensor(m_data_type, input_data)) {
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

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const UniqueParams& params) {
        const auto in = std::make_shared<op::v0::Parameter>(params.m_data_type, params.m_data_shape);
        const auto unsqueeze = std::make_shared<op::v10::Unique>(in, true, params.m_index_type);
        return std::make_shared<ov::Model>(unsqueeze, ParameterVector{in});
    }
};

TEST_P(ReferenceUniqueLayerTest_NoAxis, CompareWithHardcodedRefs) {
    Exec();
}

template <typename Data_t, typename Index_t>
std::vector<UniqueParams> generateParamsForUnique() {
    const std::vector<UniqueParams> params{UniqueParams(Shape{},
                                                        std::vector<Data_t>{1},
                                                        std::vector<Data_t>{1},
                                                        std::vector<Index_t>{0},
                                                        std::vector<Index_t>{0},
                                                        std::vector<int64_t>{1})};

    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_ReferenceUniqueLayerTest_NoAxis,
                         ReferenceUniqueLayerTest_NoAxis,
                         ::testing::ValuesIn(generateParamsForUnique<float, int32_t>()),
                         ReferenceUniqueLayerTest_NoAxis::getTestCaseName);

}  // namespace
