// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_normalization.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/group_conv.hpp"

using namespace std;
using namespace ov;
using namespace reference_tests;

namespace {
struct GroupNormalizationParams {
    GroupNormalizationParams(const reference_tests::Tensor& data,
                             const reference_tests::Tensor& scale,
                             const reference_tests::Tensor& bias,
                             const reference_tests::Tensor& expected,
                             int64_t num,
                             double eps,
                             string name)
        : data_tensor{data},
          scale_tensor{scale},
          bias_tensor{bias},
          expected_tensor{expected},
          num_groups{num},
          epsilon{eps},
          test_case_name{move(name)} {}

    reference_tests::Tensor data_tensor;
    reference_tests::Tensor scale_tensor;
    reference_tests::Tensor bias_tensor;
    reference_tests::Tensor expected_tensor;
    int64_t num_groups;
    double epsilon;
    string test_case_name;
};

class ReferenceGroupNormalization : public testing::TestWithParam<GroupNormalizationParams>,
                                    public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.data_tensor.data, params.scale_tensor.data, params.bias_tensor.data};
        refOutData = {params.expected_tensor.data};
    }

    static string getTestCaseName(const testing::TestParamInfo<GroupNormalizationParams>& obj) {
        return obj.param.test_case_name;
    }

private:
    static shared_ptr<Model> CreateFunction(const GroupNormalizationParams& params) {
        const auto in_data = make_shared<op::v0::Parameter>(params.data_tensor.type, params.data_tensor.shape);
        const auto in_scale = make_shared<op::v0::Parameter>(params.scale_tensor.type, params.scale_tensor.shape);
        const auto in_bias = make_shared<op::v0::Parameter>(params.bias_tensor.type, params.bias_tensor.shape);
        const auto group_norm =
            make_shared<op::v12::GroupNormalization>(in_data, in_scale, in_bias, params.num_groups, params.epsilon);
        return make_shared<Model>(NodeVector{group_norm}, ParameterVector{in_data, in_scale, in_bias});
    }
};

vector<GroupNormalizationParams> generateRenameMeParams() {
    vector<GroupNormalizationParams> params;

    // clang-format off
    reference_tests::Tensor data{
        {1, 4, 2, 2},
        element::f32,
        vector<float>{0.001, 0.002, 0.003, 0.004,
                      1.01, 1.02, 1.03, 1.04,
                      2.1, 2.2, 2.3, 2.4,
                      11, 12, 13, 14}};
    reference_tests::Tensor scale{{4}, element::f32, vector<float>{1, 1, 1, 1}};
    reference_tests::Tensor bias{{4}, element::f32, vector<float>{0, 0, 0, 0}};

    reference_tests::Tensor expected_4_groups{{1, 4, 2, 2},
                                              element::f32,
                                              vector<float>{-0.4472, -0.1491, 0.1491, 0.4472,
                                                            -1.2910, -0.4303, 0.4303,  1.2910,
                                                            -1.3411, -0.4470, 0.4470,  1.3411,
                                                            -1.3416, -0.4472, 0.4472,  1.3416}};
    reference_tests::Tensor expected_2_groups{{1, 4, 2, 2},
                                              element::f32,
                                              vector<float>{-1.0028, -1.0008, -0.9989, -0.9969,
                                                            0.9705,  0.9901, 1.0096,  1.0292,
                                                            -1.0171, -0.9978, -0.9786, -0.9593,
                                                            0.6990,  0.8918, 1.0846,  1.2774}};
    // clang-format on
    params.emplace_back(data, scale, bias, expected_4_groups, 4, 1e-5, "basic_4groups");
    params.emplace_back(data, scale, bias, expected_2_groups, 2, 5e-7, "basic_2groups");

    return params;
}

vector<GroupNormalizationParams> generateGroupNormalizationParams() {
    vector<vector<GroupNormalizationParams>> combo_params{generateRenameMeParams()};
    vector<GroupNormalizationParams> test_params;
    for (auto& params : combo_params)
        move(params.begin(), params.end(), back_inserter(test_params));
    return test_params;
}
}  // namespace

TEST_P(ReferenceGroupNormalization, LayerTest) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ReferenceGroupNormalization,
                         ::testing::ValuesIn(generateGroupNormalizationParams()),
                         ReferenceGroupNormalization::getTestCaseName);
