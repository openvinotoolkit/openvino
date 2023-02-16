// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iterator>

//#include "common//tests_commons/"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/manager.hpp"
#include "ops/pwl.hpp"
#include "tests_commons/library.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/pwl_approximation.hpp"

using namespace ov;
using namespace ov::opset9;
using namespace ov::element;
using namespace ov::runtime;
using namespace ov::intel_gna::op;

namespace {

// TODO create common type
using TestConfig = std::tuple<double,                          // lower_bound
                              double,                          // upper_bound
                              std::shared_ptr<Model>,          // model
                              tests_common::ReferenceFunction  // refernece function
                              >;

class PWLMonotocityTest : public ::testing::TestWithParam<TestConfig> {
public:
    static std::string GetTestCaseName(const testing::TestParamInfo<TestConfig>& obj);
    void SetUp() override;
    void Run();

private:
    static std::vector<float> run_model(std::shared_ptr<Model>& model, const TensorVector& input_tensor_vector);

    double m_lower_bound;
    double m_upper_bound;
    std::shared_ptr<Model> m_tested_model;
    tests_common::ReferenceFunction m_reference_function;
    std::vector<double> m_pwl_mes;
    std::vector<float> m_input_data;
    ov::element::Type m_precision;
    ov::Shape m_shape;
};

std::string PWLMonotocityTest::GetTestCaseName(const testing::TestParamInfo<TestConfig>& obj) {
    std::shared_ptr<Model> model;
    double lower_bound = 0.;
    double upper_bound = 0.;
    tests_common::ReferenceFunction reference_function;
    std::tie(lower_bound, upper_bound, model, reference_function) = obj.param;
    std::stringstream test_name;
    test_name << "ModelWithActivation=" << model->get_friendly_name() << "_";
    test_name << "IS=" << CommonTestUtils::vec2str(model->input().get_node_shared_ptr()->get_output_shape(0)) << "_";
    test_name << "LowerBound=" << lower_bound << "_";
    test_name << "UpperBound=" << upper_bound << "_";
    return test_name.str();
}

void PWLMonotocityTest::SetUp() {
    std::tie(m_lower_bound, m_upper_bound, m_tested_model, m_reference_function) = GetParam();
    m_pwl_mes = tests_common::generate_pwl_me_values();
    auto param = std::dynamic_pointer_cast<Parameter>(m_tested_model->input().get_node_shared_ptr());
    m_precision = param->get_element_type();
    m_shape = param->get_output_shape(0);
    m_input_data =
        tests_common::generate_input_data_covering_interal(shape_size(m_shape), m_lower_bound, m_upper_bound);
}

void PWLMonotocityTest::Run() {
    tests_common::ResultWriterFile result_writer;
    tests_common::ResultCollector result_collector(result_writer, true);

    result_collector.set_function_name(m_tested_model->get_friendly_name());

    result_collector.set_input_data(m_input_data);
    auto result_ref_function = tests_common::run_reference_function(m_reference_function, m_input_data);
    result_collector.set_reference_data(result_ref_function);

    Tensor input_tensor{m_precision, m_shape, m_input_data.data()};

    for (const auto& pwl_me : m_pwl_mes) {
        auto model_under_test = m_tested_model->clone();
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::PWLApproximation>(pwl_me);
        m.run_passes(model_under_test);
        ASSERT_NO_THROW(check_rt_info(model_under_test));

        std::vector<float> result = run_model(model_under_test, TensorVector{input_tensor});
        ASSERT_FALSE(result.empty());

        auto segments = tests_common::get_first_pwl_segments(model_under_test);
        result_collector.add_results(pwl_me, segments, result);
    }

    // TODO only for debugging purporses
    auto file_name = std::string(m_tested_model->get_friendly_name());
    file_name += "_";
    file_name += std::to_string(m_input_data.size());
    file_name += "_test_results_unit.txt";

    result_collector.store_results(file_name, tests_common::Orientation::VERTICAL);
    // TODO change validation function
    tests_common::fail_in_case_errors_are_not_deareasing_monotonically(result_collector.get_collected_data_ref());
}

std::vector<float> PWLMonotocityTest::run_model(std::shared_ptr<Model>& model,
                                                const TensorVector& input_tensor_vector) {
    TensorVector result_ref(1);
    if (!model->evaluate(result_ref, input_tensor_vector)) {
        return std::vector<float>{};
    }
    auto size = result_ref[0].get_size();
    const float* result_ref_data = result_ref[0].data<float>();
    return std::vector<float>(result_ref_data, result_ref_data + size);
}

}  // namespace

TEST_P(PWLMonotocityTest, CheckAccuracy) {
    Run();
}

// TODO both model and reference function depends on the same template type.
// There could be one creator for both.

/**
 * Tests if errors are decreasing monotonically together with decreasing max error percent parameter.
 */
INSTANTIATE_TEST_SUITE_P(
    gna_pwl_accuracy,
    PWLMonotocityTest,
    ::testing::Values(std::make_tuple(-10.,
                                      10.,
                                      tests_common::create_model<Sigmoid>(f32, Shape{1, 8}, "Sigmoid"),
                                      tests_common::ReferenceFunctionFactory<Sigmoid>::get_function())),
    // TODO add tests for missing functions.
    PWLMonotocityTest::GetTestCaseName);
