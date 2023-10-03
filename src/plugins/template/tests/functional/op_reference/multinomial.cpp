// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multinomial.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/parameter.hpp"

namespace {
struct MultinomialParams {
    MultinomialParams(const reference_tests::Tensor& probabilities,
                      const reference_tests::Tensor& num_samples,
                      const reference_tests::Tensor& expected_tensor,
                      ov::element::Type_t convert_type,
                      bool log_probs,
                      bool with_replacement,
                      std::string name)
        : probabilities{probabilities},
          num_samples{num_samples},
          expected_tensor(expected_tensor),
          convert_type{convert_type},
          log_probs(log_probs),
          with_replacement(with_replacement),
          test_case_name{std::move(name)} {}

    reference_tests::Tensor probabilities;
    reference_tests::Tensor num_samples;
    reference_tests::Tensor expected_tensor;

    ov::element::Type_t convert_type;
    bool log_probs;
    bool with_replacement;
    std::string test_case_name;
};

class ReferenceMultinomial : public testing::TestWithParam<MultinomialParams>,
                             public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.probabilities.data, params.num_samples.data};
        refOutData = {params.expected_tensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<MultinomialParams>& obj) {
        std::ostringstream name;
        name << obj.param.test_case_name;
        name << "_input_type_";
        name << obj.param.probabilities.type;
        name << "_samples_type_";
        name << obj.param.num_samples.type;
        name << "_convert_type_";
        name << obj.param.convert_type;
        name << "_log_";
        name << obj.param.log_probs;
        name << "_replacement_";
        name << obj.param.with_replacement;
        return name.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const MultinomialParams& params) {
        const auto in_probabilities =
            std::make_shared<ov::op::v0::Parameter>(params.probabilities.type, params.probabilities.shape);
        const auto in_num_samples =
            std::make_shared<ov::op::v0::Parameter>(params.num_samples.type, params.num_samples.shape);
        const auto multinomial = std::make_shared<ov::op::v13::Multinomial>(in_probabilities,
                                                                            in_num_samples,
                                                                            params.convert_type,
                                                                            params.with_replacement,
                                                                            params.log_probs,
                                                                            1,
                                                                            1);
        return std::make_shared<ov::Model>(multinomial->outputs(),
                                           ov::ParameterVector{in_probabilities, in_num_samples});
    }
};

template <ov::element::Type_t et>
std::vector<MultinomialParams> generateMultinomialParams() {
    using vt = typename ov::element_type_traits<et>::value_type;

    const ov::Shape prob_2d_shape{2, 4};
    const ov::Shape prob_1d_shape{4};
    const ov::Shape num_samples_shape{1};

    reference_tests::Tensor num_samples(num_samples_shape, ov::element::Type_t::i32, std::vector<int32_t>{4});

    reference_tests::Tensor probabilities_2d_no_log(prob_2d_shape,
                                                    et,
                                                    std::vector<vt>{0.001, 0.01, 0.1, 0.899, 0.899, 0.1, 0.01, 0.001});
    reference_tests::Tensor probabilities_2d_log(prob_2d_shape, et, std::vector<vt>{1, 2, 3, 4, 2, 4, 6, 8});
    reference_tests::Tensor probabilities_1d_no_log(prob_1d_shape, et, std::vector<vt>{0.001, 0.01, 0.1, 0.899});
    reference_tests::Tensor probabilities_1d_log(prob_1d_shape, et, std::vector<vt>{1, 10, 7, 3});

    reference_tests::Tensor output_2d_no_log_no_replacement(prob_2d_shape,
                                                            ov::element::Type_t::i32,
                                                            std::vector<int32_t>{3, 3, 3, 3, 0, 0, 0, 0});
    reference_tests::Tensor output_2d_log_no_replacement(prob_2d_shape,
                                                         ov::element::Type_t::i32,
                                                         std::vector<int32_t>{3, 3, 2, 3, 3, 3, 3, 3});
    reference_tests::Tensor output_1d_no_log_replacement(prob_1d_shape,
                                                         ov::element::Type_t::i64,
                                                         std::vector<int64_t>{3, 2, 1, 0});
    reference_tests::Tensor output_1d_log_replacement(prob_1d_shape,
                                                      ov::element::Type_t::i64,
                                                      std::vector<int64_t>{1, 2, 3, 0});

    std::vector<MultinomialParams> params;
    // probabilities, num_samples, output, convert_type, log_probs, with_replacement, name
    params.emplace_back(probabilities_2d_no_log,
                        num_samples,
                        output_2d_no_log_no_replacement,
                        ov::element::Type_t::i32,
                        false,
                        false,
                        "input_2d");
    params.emplace_back(probabilities_2d_log,
                        num_samples,
                        output_2d_log_no_replacement,
                        ov::element::Type_t::i32,
                        true,
                        false,
                        "input_2d");
    params.emplace_back(probabilities_1d_no_log,
                        num_samples,
                        output_1d_no_log_replacement,
                        ov::element::Type_t::i64,
                        false,
                        true,
                        "input_1d");
    params.emplace_back(probabilities_1d_log,
                        num_samples,
                        output_1d_log_replacement,
                        ov::element::Type_t::i64,
                        true,
                        true,
                        "input_1d");
    return params;
}

std::vector<MultinomialParams> generateMultinomialParams() {
    std::vector<std::vector<MultinomialParams>> combo_params{generateMultinomialParams<ov::element::f32>()};
    std::vector<MultinomialParams> test_params;
    for (auto& params : combo_params)
        std::move(params.begin(), params.end(), std::back_inserter(test_params));
    return test_params;
}
}  // namespace

TEST_P(ReferenceMultinomial, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ReferenceMultinomial,
                         ::testing::ValuesIn(generateMultinomialParams()),
                         ReferenceMultinomial::getTestCaseName);
