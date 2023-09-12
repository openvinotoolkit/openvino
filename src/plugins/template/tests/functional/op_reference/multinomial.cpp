// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/group_normalization.hpp"

using namespace std;
using namespace ov;
using namespace reference_tests;

namespace {
struct MultinomialParams {
    MultinomialParams(const reference_tests::Tensor& probabilities,
                      const reference_tests::Tensor& num_samples,
                      const reference_tests::Tensor& expected_tensor,
                      ngraph::element::Type_t output_type,
                      bool log_probs,
                      bool with_replacement,
                      string name)
        : probabilities{probabilities},
          num_samples{num_samples},
          expected_tensor(expected_tensor),
          output_type{output_type},
          test_case_name{std::move(name)} {}

    reference_tests::Tensor probabilities;
    reference_tests::Tensor num_samples;
    reference_tests::Tensor expected_tensor;

    ngraph::element::Type_t output_type;
    bool log_probs;
    bool with_replacement;
    string test_case_name;
};

class ReferenceMultinomial : public testing::TestWithParam<MultinomialParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.probabilities.data, params.num_samples.data};
        refOutData = {params.expected_tensor.data};
    }

    static string getTestCaseName(const testing::TestParamInfo<MultinomialParams>& obj) {
        std::ostringstream name;
        name << obj.param.test_case_name;
        name << "_input_type_";
        name << obj.param.probabilities.type;
        name << "_samples_type_";
        name << obj.param.num_samples.type;
        name << "_output_type_";
        name << obj.param.output_type;
        name << "_log_";
        name << obj.param.log_probs;
        name << "_replacement_";
        name << obj.param.with_replacement;
        return name.str();
    }

private:
    static shared_ptr<Model> CreateFunction(const MultinomialParams& params) {
        const auto in_probabilities =
            make_shared<op::v0::Parameter>(params.probabilities.type, params.probabilities.shape);
        const auto in_num_samples = make_shared<op::v0::Parameter>(params.num_samples.type, params.num_samples.shape);
        const auto multinomial = make_shared<op::v13::Multinomial>(in_probabilities,
                                                                   in_num_samples,
                                                                   params.with_replacement,
                                                                   params.log_probs,
                                                                   0,
                                                                   0);
        return make_shared<Model>(NodeVector{multinomial}, ParameterVector{in_probabilities, in_num_samples});
    }
};

template <element::Type_t et>
vector<MultinomialParams> generateMultinomialParams() {
    using vt = typename element_type_traits<et>::value_type;

    const Shape prob_2d_shape{2, 4};
    const Shape prob_1d_shape{4};
    const Shape num_samples_shape{1};

    reference_tests::Tensor num_samples{num_samples_shape, et, vector<int32_t>{4}};

    reference_tests::Tensor probabilities_2d_no_log{prob_2d_shape,
                                                    et,
                                                    vector<vt>{0.001, 0.01, 0.1, 0.899, 0.899, 0.1, 0.01, 0.001}};
    reference_tests::Tensor probabilities_2d_log{prob_2d_shape, et, vector<vt>{1, 10, 100, 10000, 2, 20, 200, 20000}};

    reference_tests::Tensor probabilities_1d_no_log{prob_1d_shape, et, vector<vt>{0.001, 0.01, 0.1, 0.899}};
    reference_tests::Tensor probabilities_1d_log{prob_1d_shape, et, vector<vt>{1, 10000, 100, 10}};

    reference_tests::Tensor output_2d_no_log_no_replacement{prob_2d_shape, et, vector<vt>{3, 2, 1, 0, 4, 5, 6, 7}};
    reference_tests::Tensor output_2d_log_no_replacement{prob_2d_shape, et, vector<vt>{3, 2, 1, 0, 7, 6, 5, 4}};
    reference_tests::Tensor output_1d_no_log_replacement{prob_1d_shape, et, vector<vt>{3, 3, 3, 3}};
    reference_tests::Tensor output_1d_log_replacement{prob_1d_shape, et, vector<vt>{2, 2, 2, 2}};

    vector<MultinomialParams> params;
    // probabilities, num_samples, output, out_type, log_probs, with_replacement, name
    params.emplace_back(probabilities_2d_no_log,
                        num_samples,
                        output_2d_no_log_no_replacement,
                        vt,
                        false,
                        false,
                        "input_2d");
    params.emplace_back(probabilities_2d_log, num_samples, output_2d_log_no_replacement, vt, true, false, "input_2d");
    params
        .emplace_back(probabilities_1d_no_log, num_samples, output_1d_no_log_replacement, vt, false, true, "input_1d");
    params.emplace_back(probabilities_1d_log, num_samples, output_1d_log_replacement, vt, true, true, "input_1d");
    return params;
}

vector<MultinomialParams> generateMultinomialParams() {
    vector<vector<MultinomialParams>> combo_params{generateMultinomialParams<element::f16>(),
                                                   generateMultinomialParams<element::f32>(),
                                                   generateMultinomialParams<element::f64>()};
    vector<MultinomialParams> test_params;
    for (auto& params : combo_params)
        move(params.begin(), params.end(), back_inserter(test_params));
    return test_params;
}
}  // namespace

TEST_P(ReferenceMultinomial, LayerTest) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ReferenceMultinomial,
                         ::testing::ValuesIn(generateMultinomialParams()),
                         ReferenceMultinomial::getTestCaseName);
