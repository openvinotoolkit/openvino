// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iterator>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/pwl_approximation.hpp"

using namespace ov;
using namespace ov::opset9;
using namespace ov::element;
using namespace ov::runtime;

namespace {

struct Error {
    double max_error;
    double sum_of_errors;
    size_t num_of_errors;

    double avg_error() const {
        if (num_of_errors == 0) {
            return 0;
        }
        return sum_of_errors / num_of_errors;
    }
};

std::ostream& operator<<(std::ostream& os, const Error& error) {
    os << "Error(" << std::setprecision(15) << " max_error(" << error.max_error << "), avg_error(" << error.avg_error()
       << "))";
    return os;
}

class ErrorValidator {
public:
    static Error CountError(const std::vector<float>& expected_buffer, const std::vector<float>& actual_buffer) {
        Error error = {0.0, 0.0, 0};
        for (int i = 0; i < expected_buffer.size(); ++i) {
            double error_val = std::abs(expected_buffer[i] - actual_buffer[i]);
            error.sum_of_errors += error_val;
            if (error_val > error.max_error) {
                error.max_error = error_val;
            }
            error.num_of_errors++;
        }
        return error;
    }

    static bool IsPrevErrorLower(const Error& prev_error, const Error& current_error) {
        if (prev_error.max_error < current_error.max_error || prev_error.avg_error() < current_error.avg_error()) {
            return true;
        }

        return false;
    }

    /**
     * Return empty string if current_pwl_error is lower than previous_pwl_error.
     */
    static std::string ValidateErrorCurrentLowerThanPrev(const std::pair<double, Error>& current_pwl_error,
                                                  const std::pair<double, Error>& previous_pwl_error) {
        if (IsPrevErrorLower(previous_pwl_error.second, current_pwl_error.second)) {
            std::stringstream sstr;
            sstr << "pwl_me[" << current_pwl_error.first << "]: Error is bigger than for previous pwl_me["
                 << previous_pwl_error.first << "]. ";
            sstr << "Current (" << current_pwl_error.second << "), Previous (" << previous_pwl_error.second << ")"
                 << std::endl;
            return sstr.str();
        }
        return "";
    }
};

using ActivationCreator = std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>;

template <typename T>
ActivationCreator CreateActivationCreator() {
    return [](std::shared_ptr<Node> input_layer) {
        return std::make_shared<T>(input_layer);
    };
}

template <typename T>
ActivationCreator CreateActivationCreator(double arg) {
    return [arg](std::shared_ptr<Node> input_layer) {
        auto exponents = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {arg});
        return std::make_shared<T>(input_layer, exponents);
    };
}

std::shared_ptr<Model> CreateModel(ov::element::Type precision,
                                   const Shape& input_shape,
                                   const std::string model_name,
                                   ActivationCreator activation_creator) {
    auto input_params = std::make_shared<Parameter>(precision, input_shape);

    const auto multiply_const_size = shape_size(input_shape);
    const std::vector<float> multiply_const_data(multiply_const_size, 1.);
    auto multiply_const = std::make_shared<Constant>(precision, input_shape, multiply_const_data);
    auto multiply = std::make_shared<Multiply>(input_params, multiply_const);
    auto activation = activation_creator(multiply);
    auto result = std::make_shared<Result>(activation);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{input_params}, model_name);
}

using TestConfig = std::tuple<double,                 // lower_bound
                              double,                 // upper_bound
                              std::shared_ptr<Model>  // model
                              >;

class PWLMonotocityTest : public ::testing::TestWithParam<TestConfig> {
public:
    static std::string GetTestCaseName(const testing::TestParamInfo<TestConfig>& obj);
    void SetUp() override;
    void Run();

private:
    std::vector<double> GeneratePWLMEValues();
    std::vector<float> GenerateInput(Type precision, Shape& shape);
    std::vector<float> RunModel(std::shared_ptr<Model>& model, const TensorVector& input_tensor_vector);

    void ValidatePwlMeErrors(const std::vector<std::pair<double, Error>>& errors);

    double m_lower_bound;
    double m_upper_bound;
    std::shared_ptr<Model> m_tested_model;
};

std::string PWLMonotocityTest::GetTestCaseName(const testing::TestParamInfo<TestConfig>& obj) {
    std::shared_ptr<Model> model;
    double lower_bound = 0.;
    double upper_bound = 0.;
    std::tie(lower_bound, upper_bound, model) = obj.param;
    std::stringstream test_name;
    test_name << "ModelWithActivation=" << model->get_friendly_name() << "_";
    test_name << "IS=" << CommonTestUtils::vec2str(model->input().get_node_shared_ptr()->get_output_shape(0)) << "_";
    test_name << "LowerBound=" << lower_bound << "_";
    test_name << "UpperBound=" << upper_bound << "_";
    return test_name.str();
}

void PWLMonotocityTest::SetUp() {
    std::tie(m_lower_bound, m_upper_bound, m_tested_model) = GetParam();
}

void PWLMonotocityTest::Run() {
    auto param = std::dynamic_pointer_cast<Parameter>(m_tested_model->input().get_node_shared_ptr());
    auto precision = param->get_element_type();
    auto shape = param->get_output_shape(0);

    auto input_data = GenerateInput(precision, shape);
    Tensor input_tensor{precision, shape, input_data.data()};

    std::vector<float> result_ref = RunModel(m_tested_model, TensorVector{input_tensor});

    ASSERT_FALSE(result_ref.empty());

    std::vector<std::pair<double, Error>> pwl_me_errors;
    auto pwl_mes = GeneratePWLMEValues();

    for (const auto& pwl_me : pwl_mes) {
        auto model_under_test = m_tested_model->clone();
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::PWLApproximation>(pwl_me);
        m.run_passes(model_under_test);
        ASSERT_NO_THROW(check_rt_info(model_under_test));

        std::vector<float> result = RunModel(model_under_test, TensorVector{input_tensor});
        ASSERT_FALSE(result.empty());

        auto error = ErrorValidator::CountError(result_ref, result);

        pwl_me_errors.emplace_back(pwl_me, std::move(error));
    }
    ValidatePwlMeErrors(pwl_me_errors);
}

/**
 * Generate pwls.
 */
std::vector<double> PWLMonotocityTest::GeneratePWLMEValues() {
    std::vector<double> pwl_mes;
    // add 1.0 to 0.1 step 0.1
    for (int i = 10; i > 0; --i) {
        pwl_mes.push_back(i / 10.);
    }

    // add 0.09 to 0.03 step 0.01
    for (int i = 9; i > 3; --i) {
        pwl_mes.push_back(i / 100.);
    }
    return pwl_mes;
}

/**
 * Generates inputs from lower_bound to upper_bound with step dependent on shape.
 */
std::vector<float> PWLMonotocityTest::GenerateInput(Type precision, Shape& shape) {
    auto number_of_data = shape_size(shape);
    std::vector<float> data(number_of_data, 1);
    float step = (m_upper_bound - m_lower_bound) / (number_of_data - 1);
    float value = m_lower_bound;
    for (size_t i = 0; i < number_of_data; i++) {
        data[i] = value;
        value += step;
    }
    return data;
}

std::vector<float> PWLMonotocityTest::RunModel(std::shared_ptr<Model>& model,
                                                  const TensorVector& input_tensor_vector) {
    TensorVector result_ref(1);
    if (!model->evaluate(result_ref, input_tensor_vector)) {
        return std::vector<float>{};
    }
    auto size = result_ref[0].get_size();
    const float* result_ref_data = result_ref[0].data<float>();
    return std::vector<float>(result_ref_data, result_ref_data + size);
}

void PWLMonotocityTest ::ValidatePwlMeErrors(const std::vector<std::pair<double, Error>>& errors) {
    auto prev_error = errors[0];
    for (int i = 1; i < errors.size(); ++i) {
        auto error_message = ErrorValidator::ValidateErrorCurrentLowerThanPrev(errors[i], prev_error);
        if (!error_message.empty()) {
            FAIL() << error_message;
        }
        prev_error = errors[i];
    }
}

}  // namespace

TEST_P(PWLMonotocityTest, CheckAccuracy) {
    Run();
}
/**
 * Tests if errors are decreasing monotonically together with decreasing max error percent parameter.
 */
INSTANTIATE_TEST_SUITE_P(
    gna_pwl_accuracy,
    PWLMonotocityTest,
    ::testing::Values(
        std::make_tuple(-10., 10., CreateModel(f32, Shape{1, 128}, "Sigmoid", CreateActivationCreator<Sigmoid>())),
        std::make_tuple(-10., 10., CreateModel(f32, Shape{1, 8}, "Sigmoid", CreateActivationCreator<Sigmoid>())),
        std::make_tuple(-5., 5., CreateModel(f32, Shape{1, 128}, "Tanh", CreateActivationCreator<Tanh>())),
        std::make_tuple(-5., 5., CreateModel(f32, Shape{1, 8}, "Tanh", CreateActivationCreator<Tanh>())),
        std::make_tuple(-5., 5., CreateModel(f32, Shape{1, 128}, "Exp", CreateActivationCreator<Exp>())),
        std::make_tuple(-5., 5., CreateModel(f32, Shape{1, 8}, "Exp", CreateActivationCreator<Exp>())),
        std::make_tuple(-10., 10, CreateModel(f32, Shape{1, 128}, "SoftSign", CreateActivationCreator<SoftSign>())),
        std::make_tuple(-10., 10, CreateModel(f32, Shape{1, 8}, "SoftSign", CreateActivationCreator<SoftSign>())),
        std::make_tuple(0.001, 2981, CreateModel(f32, Shape{1, 128}, "Log", CreateActivationCreator<Log>())),
        std::make_tuple(0.001, 2981, CreateModel(f32, Shape{1, 8}, "Log", CreateActivationCreator<Log>())),
        std::make_tuple(0, 16, CreateModel(f32, Shape{1, 128}, "Power", CreateActivationCreator<Power>(2.0))),
        std::make_tuple(0, 16, CreateModel(f32, Shape{1, 8}, "Power", CreateActivationCreator<Power>(2.0)))),
    PWLMonotocityTest::GetTestCaseName);
