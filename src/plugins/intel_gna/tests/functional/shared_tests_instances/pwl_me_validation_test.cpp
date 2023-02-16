// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "tests_commons/library.hpp"

namespace LayerTestsDefinitions {
using TestConfig = std::tuple<double,                          // lower_bound
                              double,                          // upper_bound
                              std::shared_ptr<ov::Model>,      // model
                              tests_common::ReferenceFunction  // refernece function
                              >;

class PWLApproxmiationPWLMeTest : public testing::WithParamInterface<TestConfig>,
                                  public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TestConfig> obj);

protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;
    void SetUp() override;
    void Run() override;

    std::vector<float> retrieve_results();
    template <typename T>
    static tests_common::Error count_error_per_output(const T* expected_buffer, const T* actual_buffer, size_t size);
    static void validate_pwl_me_errors(const std::vector<std::pair<std::string, tests_common::Error>>& errors);

    /*
     * Return empty string if current_pwl_error is lower than prvious_pwl_error, otherwise return error string.
     */
    static std::string validate_error_current_lower_than_prev(
        const std::pair<std::string, tests_common::Error>& current_pwl_error,
        const std::pair<std::string, tests_common::Error>& prvious_pwl_error);

private:
    float m_upper_bound = 1.0;
    float m_lower_bound = -1.0;
    tests_common::ReferenceFunction m_reference_function;
    std::vector<double> m_pwl_mes;
    std::vector<float> m_input_data;
    ov::element::Type m_precision;
    ov::Shape m_shape;
};

TEST_P(PWLApproxmiationPWLMeTest, CompareWithRefImpl) {
    Run();
};

std::string PWLApproxmiationPWLMeTest::getTestCaseName(testing::TestParamInfo<TestConfig> obj) {
    std::shared_ptr<ov::Model> model;
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

InferenceEngine::Blob::Ptr PWLApproxmiationPWLMeTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();

    auto* raw_blob_data_ptr = blob->buffer().as<float*>();

    std::copy(m_input_data.begin(), m_input_data.end(), raw_blob_data_ptr);

    return blob;
}

void PWLApproxmiationPWLMeTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_GNA;

    std::tie(m_lower_bound, m_upper_bound, function, m_reference_function) = GetParam();
    m_pwl_mes = tests_common::generate_pwl_me_values();
    auto param = std::dynamic_pointer_cast<ov::opset9::Parameter>(function->input().get_node_shared_ptr());
    m_precision = param->get_element_type();
    m_shape = param->get_output_shape(0);
    m_input_data =
        tests_common::generate_input_data_covering_interal(shape_size(m_shape), m_lower_bound, m_upper_bound);
}

void PWLApproxmiationPWLMeTest::Run() {
    tests_common::ResultWriterFile result_writer;
    tests_common::ResultCollector result_collector(result_writer, true);

    result_collector.set_function_name(function->get_friendly_name());
    result_collector.set_input_data(m_input_data);
    auto result_ref_function = tests_common::run_reference_function(m_reference_function, m_input_data);
    result_collector.set_reference_data(result_ref_function);

    for (const auto& pwl_me : m_pwl_mes) {
        configuration = {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_PWL_MAX_ERROR_PERCENT", std::to_string(pwl_me)}};

        LoadNetwork();
        GenerateInputs();
        Infer();
        std::vector<float> result = retrieve_results();
        result_collector.add_results(pwl_me, {}, result);
    }

    auto file_name = std::string(function->get_friendly_name());
    file_name += "_";
    file_name += std::to_string(m_input_data.size());
    file_name += "_test_results_func.txt";
    result_collector.store_results(file_name, tests_common::Orientation::VERTICAL);

    // TODO change validation condition.
    tests_common::fail_in_case_errors_are_not_deareasing_monotonically(result_collector.get_collected_data_ref());
}

std::vector<float> PWLApproxmiationPWLMeTest::retrieve_results() {
    const auto& actual_outputs = GetOutputs();

    IE_ASSERT(actual_outputs.size() == 1) << "Unsupported outputs number. Model should have one output";

    const auto& actual = actual_outputs[0];

    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
    IE_ASSERT(memory);
    const auto locked_memory = memory->wmap();

    auto precision = actual->getTensorDesc().getPrecision();

    // TODO double check - warning
    if (!precision == InferenceEngine::Precision::FP32) {
        IE_THROW() << "Precision: " << precision << " precision isn't supported";
    }
    const auto actual_output_buffer = locked_memory.as<const float*>();

    return {actual_output_buffer, actual_output_buffer + actual->size()};
}

INSTANTIATE_TEST_SUITE_P(
    smoke_base_pwl_me_sigmoid,
    PWLApproxmiationPWLMeTest,
    ::testing::Values(
        std::make_tuple(-10.,
                        10.0,
                        tests_common::create_model<ov::opset9::Sigmoid>(ov::element::f32, ov::Shape{1, 128}, "Sigmoid"),
                        tests_common::ReferenceFunctionFactory<ov::opset9::Sigmoid>::get_function()),
        std::make_tuple(-5.,
                        5.0,
                        tests_common::create_model<ov::opset9::Tanh>(ov::element::f32, ov::Shape{1, 128}, "Tanh"),
                        tests_common::ReferenceFunctionFactory<ov::opset9::Tanh>::get_function())),
    PWLApproxmiationPWLMeTest::getTestCaseName);

// TODO add missing implementation for
// SoftSign, Exp, Power, Power, Log
// TODO Create Models
//    * separate for Multiply and each activation function
//    * common with multiply and all activation functions

}  // namespace LayerTestsDefinitions