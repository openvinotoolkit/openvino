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

// TODO benchmarking
struct Timer {
    Timer(const std::string& name,
          bool print_on_destruction = false,
          std::shared_ptr<std::chrono::duration<double, std::micro>> time_spent = nullptr)
        : _name(name),
          _print_on_destruction(print_on_destruction),
          _time_spent(time_spent) {
        _start_time = std::chrono::steady_clock::now();
    }

public:
    double get_time_spent_in_us() {
        return get_time_spent().count();
    }
    std::chrono::duration<double, std::micro> get_time_spent() {
        return std::chrono::steady_clock::now() - _start_time;
    }
    void print_time_spent() {
        std::cout << "[" << _name << "]: " << get_time_spent_in_us() << " us" << std::endl;
        auto current = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> time_spent = current - _start_time;
    }
    ~Timer() {
        auto time_spent = get_time_spent();
        if (_time_spent) {
            *_time_spent += time_spent;
        }
        if (_print_on_destruction) {
            std::cout << "[" << _name << "]: " << time_spent.count() << " us" << std::endl;
            if (_time_spent) {
                std::cout << "[" << _name << "][TOTAL]: " << _time_spent->count() << " us" << std::endl;
            }
        }
    }
    std::string _name;
    std::chrono::time_point<std::chrono::steady_clock> _start_time;
    std::shared_ptr<std::chrono::duration<double, std::micro>> _time_spent;
    bool _print_on_destruction;
};

// TODO benchmarking
void print_func(std::string operation, float input_min, float input_max) {
    std::cout << "[ACT]: " << operation << std::endl;
    std::cout << "[INPUT_MIN]: " << input_min << std::endl;
    std::cout << "[INPUT_MAX]: " << input_max << std::endl;
    std::cout << std::endl;
}

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

bool is_prev_error_lower(const Error& prev_error, const Error& current_error) {
    if (prev_error.max_error < current_error.max_error || prev_error.avg_error() < current_error.avg_error()) {
        return true;
    }

    return false;
}

std::ostream& operator<<(std::ostream& os, const Error& error) {
    os << "Error(" << std::setprecision(15) << " max_error(" << error.max_error << "), avg_error(" << error.avg_error()
       << "))";
    return os;
}

enum class PWLApproximationOp { Sigmoid, Tanh, Exp, Power, Log, SoftSign };

static std::map<PWLApproximationOp, std::string> operations_names = {{PWLApproximationOp::Sigmoid, "Sigmoid"},
                                                                     {PWLApproximationOp::Tanh, "Tanh"},
                                                                     {PWLApproximationOp::Exp, "Exp"},
                                                                     {PWLApproximationOp::Power, "Power"},
                                                                     {PWLApproximationOp::Log, "Log"},
                                                                     {PWLApproximationOp::SoftSign, "SoftSign"}};

std::shared_ptr<ngraph::Node> make_function(const ngraph::Output<ngraph::Node>& in,
                                            PWLApproximationOp operation,
                                            ::ngraph::element::Type precision,
                                            float optional = 0.0) {
    switch (operation) {
    case PWLApproximationOp::Sigmoid:
        return std::make_shared<ngraph::opset9::Sigmoid>(in);
    case PWLApproximationOp::Tanh:
        return std::make_shared<ngraph::opset9::Tanh>(in);
    case PWLApproximationOp::Exp:
        return std::make_shared<ngraph::opset9::Exp>(in);
    case PWLApproximationOp::Power: {
        auto exponents = ngraph::opset9::Constant::create(precision, ngraph::Shape{}, {optional});
        return std::make_shared<ngraph::opset9::Power>(in, exponents);
    }
    case PWLApproximationOp::Log:
        return std::make_shared<ngraph::opset9::Log>(in);
    case PWLApproximationOp::SoftSign:
        return std::make_shared<ngraph::opset9::SoftSign>(in);

    default:
        throw "Unexpected Test Value";
    }
}

typedef std::tuple<InferenceEngine::Precision,  // Network Precision
                   std::pair<float, float>,     // Input values
                   PWLApproximationOp,          // PWLApproximationOp
                   float                        // optional used by exp
                   >
    pwlAproximationFqParams;

namespace LayerTestsDefinitions {

class PWLApproxmiationPWLMeTest : public testing::WithParamInterface<pwlAproximationFqParams>,
                                  public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<pwlAproximationFqParams> obj);

protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;
    void SetUp() override;
    void Run() override;

    void validate_results_end_cound_error(Error& error);
    template <typename T>
    static Error count_error_per_output(const T* expected_buffer, const T* actual_buffer, size_t size);
    static void validate_pwl_me_errors(const std::vector<std::pair<std::string, Error>>& errors);

    /*
     * Return empty string if current_pwl_error is lower than prvious_pwl_error, otherwise return error string.
     */
    static std::string validate_error_current_lower_than_prev(const std::pair<std::string, Error>& current_pwl_error,
                                                              const std::pair<std::string, Error>& prvious_pwl_error);

    float _input_data_max = 1.0;
    float _input_data_min = -1.0;

    // TODO benchmarking
    std::string _operation_name;
};

TEST_P(PWLApproxmiationPWLMeTest, CompareWithRefImpl) {
    Run();
};

std::string PWLApproxmiationPWLMeTest::getTestCaseName(testing::TestParamInfo<pwlAproximationFqParams> obj) {
    InferenceEngine::Precision net_precision;
    std::pair<float, float> input_values;
    PWLApproximationOp operation;
    float exp = 0.0f;
    std::tie(net_precision, input_values, operation, exp) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << net_precision.name() << "_";
    result << "_range=(" << input_values.first << ", " << input_values.second << ")";
    result << "_act=" << operations_names[operation];

    return result.str();
}

InferenceEngine::Blob::Ptr PWLApproxmiationPWLMeTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();

    auto* rawBlobDataPtr = blob->buffer().as<float*>();
    float step = (_input_data_max - _input_data_min) / (blob->size() - 1);
    float value = _input_data_min;
    for (size_t i = 0; i < blob->size(); i++) {
        rawBlobDataPtr[i] = value;
        value += step;
    }
    return blob;
}

void PWLApproxmiationPWLMeTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_GNA;

    InferenceEngine::Precision net_precision;
    std::pair<float, float> input_values;
    PWLApproximationOp operation;
    float exp = 0.0f;
    std::tie(net_precision, input_values, operation, exp) = this->GetParam();
    std::tie(_input_data_min, _input_data_max) = input_values;

    // TODO benchmarking
    _operation_name = operations_names[operation];

    auto ng_prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(net_precision);

    const ngraph::Shape shape = {1, 128};
    auto params = ngraph::builder::makeParams(ng_prc, {shape});

    std::vector<float> vector_data(128, 1.0);
    auto constant = std::make_shared<ngraph::opset9::Constant>(ng_prc, shape, vector_data);
    auto multiply = std::make_shared<ngraph::opset9::Multiply>(params[0], constant);

    auto activation = make_function(multiply, operation, ng_prc, exp);

    ngraph::ResultVector results{std::make_shared<ngraph::opset9::Result>(activation)};
    function = std::make_shared<ngraph::Function>(results, params, "ActivityFq");
}

void PWLApproxmiationPWLMeTest::Run() {
    // TODO benchmarking
    print_func(_operation_name, _input_data_min, _input_data_max);

    // execute network for different pwl_me options
    std::vector<std::string> pwl_mes;
    // add 1.0 to 0.1 step 0.1
    for (int i = 10; i > 0; --i) {
        pwl_mes.push_back(std::to_string(i / 10.));
    }

    // add 0.09 to 0.02 step 0.01
    for (int i = 9; i > 1; --i) {
        pwl_mes.push_back(std::to_string(i / 100.));
    }

    std::vector<std::pair<std::string, Error>> pwl_me_errors;

    for (const auto& pwl_me : pwl_mes) {
        // TODO benchmarking
        std::cout << pwl_me << ";";

        configuration = {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_PWL_MAX_ERROR_PERCENT", pwl_me}};
        // TODO benchmarking
        Timer load_time("LOAD_TIME", false);

        LoadNetwork();
        // TODO benchmarking
        std::cout << load_time.get_time_spent_in_us() << ";";

        GenerateInputs();
        Infer();

        Error error = {};
        validate_results_end_cound_error(error);

        // TODO benchmarking
        std::cout << error.max_error << ";";
        std::cout << error.avg_error();

        pwl_me_errors.emplace_back(pwl_me, std::move(error));
        // TODO benchmarking
        std::cout << std::endl;
    }

    validate_pwl_me_errors(pwl_me_errors);
}

void PWLApproxmiationPWLMeTest::validate_results_end_cound_error(Error& error) {
    if (functionRefs == nullptr) {
        functionRefs = ngraph::clone_function(*function);
    }

    auto expected_outputs = CalculateRefs();
    const auto& actual_outputs = GetOutputs();

    IE_ASSERT(actual_outputs.size() == expected_outputs.size())
        << "nGraph interpreter has " << expected_outputs.size() << " outputs, while IE " << actual_outputs.size();

    IE_ASSERT(!actual_outputs.empty()) << "List of results is empty";

    std::vector<Error> errors;
    for (std::size_t outputIndex = 0; outputIndex < expected_outputs.size(); ++outputIndex) {
        const auto& expected = expected_outputs[outputIndex];
        const auto& actual = actual_outputs[outputIndex];

        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
        IE_ASSERT(memory);
        const auto locked_memory = memory->wmap();

        auto precision = actual->getTensorDesc().getPrecision();

        if (precision == InferenceEngine::Precision::FP16) {
            const auto actual_output_buffer = locked_memory.as<const ngraph::float16*>();
            const auto expected_output_buffer = reinterpret_cast<const ngraph::float16*>(expected.second.data());
            errors.push_back(count_error_per_output(expected_output_buffer, actual_output_buffer, actual->size()));
        } else if (precision == InferenceEngine::Precision::FP32) {
            const auto actual_output_buffer = locked_memory.as<const float*>();
            const auto expected_output_buffer = reinterpret_cast<const float*>(expected.second.data());
            errors.push_back(count_error_per_output(expected_output_buffer, actual_output_buffer, actual->size()));
        } else {
            IE_THROW() << "Comparator for " << precision << " precision isn't supported";
        }
    }

    Error temp_error = errors[0];
    for (int i = 1; i < errors.size(); i++) {
        if (temp_error.max_error < errors[i].max_error) {
            temp_error.max_error = errors[i].max_error;
        }
        temp_error.num_of_errors += errors[i].num_of_errors;
    }

    error = temp_error;
}

template <typename T>
Error PWLApproxmiationPWLMeTest::count_error_per_output(const T* expected_buffer, const T* actual_buffer, size_t size) {
    Error error = {0.0, 0.0, 0};
    for (int i = 0; i < size; ++i) {
        double error_val = std::abs(expected_buffer[i] - actual_buffer[i]);
        error.sum_of_errors += error_val;
        if (error_val > error.max_error) {
            error.max_error = error_val;
        }
        error.num_of_errors++;
    }
    return error;
}

void PWLApproxmiationPWLMeTest::validate_pwl_me_errors(const std::vector<std::pair<std::string, Error>>& errors) {
    auto prev_error = errors[0];
    for (int i = 1; i < errors.size(); ++i) {
        auto error_message = validate_error_current_lower_than_prev(errors[i], prev_error);
        if (!error_message.empty()) {
            FAIL() << error_message;
        }
        prev_error = errors[i];
    }
}

/*
 * Return empty string if current_pwl_error is lower than prvious_pwl_error.
 */
std::string PWLApproxmiationPWLMeTest::validate_error_current_lower_than_prev(
    const std::pair<std::string, Error>& current_pwl_error,
    const std::pair<std::string, Error>& prvious_pwl_error) {
    if (is_prev_error_lower(prvious_pwl_error.second, current_pwl_error.second)) {
        std::stringstream sstr;
        sstr << "pwl_me[" << current_pwl_error.first << "]: Error is bigger than for prevois pwl_me["
             << prvious_pwl_error.first << "]. ";
        sstr << "Current (" << current_pwl_error.second << "), Previous (" << prvious_pwl_error.second << ")"
             << std::endl;
        return sstr.str();
    }
    return "";
}

const std::vector<InferenceEngine::Precision> net_precisions = {InferenceEngine::Precision::FP32};

const std::vector<std::pair<float, float>> input_values_ranges = {{-10.0, 10.0}, {-0.04, 0.04}};
const std::vector<std::pair<float, float>> input_values_ranges_log = {{1.0e-3, 10.0}, {1.0e-3, 0.04}};

// Test if accuracy is not decreasing when pwl_me value is decreasing for various activation functions

INSTANTIATE_TEST_SUITE_P(smoke_base_pwl_me_sigmoid,
                         PWLApproxmiationPWLMeTest,
                         ::testing::Combine(::testing::ValuesIn(net_precisions),
                                            ::testing::ValuesIn(input_values_ranges),
                                            ::testing::Values(PWLApproximationOp::Sigmoid),
                                            ::testing::Values(0.0f)),
                         PWLApproxmiationPWLMeTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_base_pwl_me_tanh,
                         PWLApproxmiationPWLMeTest,
                         ::testing::Combine(::testing::ValuesIn(net_precisions),
                                            ::testing::ValuesIn(input_values_ranges),
                                            ::testing::Values(PWLApproximationOp::Tanh),
                                            ::testing::Values(0.0f)),
                         PWLApproxmiationPWLMeTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_base_pwl_me_exp,
                         PWLApproxmiationPWLMeTest,
                         ::testing::Combine(::testing::ValuesIn(net_precisions),
                                            ::testing::ValuesIn(input_values_ranges),
                                            ::testing::Values(PWLApproximationOp::Exp),
                                            ::testing::Values(0.0f)),
                         PWLApproxmiationPWLMeTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_base_pwl_me_power,
                         PWLApproxmiationPWLMeTest,
                         ::testing::Combine(::testing::ValuesIn(net_precisions),
                                            ::testing::ValuesIn(input_values_ranges),
                                            ::testing::Values(PWLApproximationOp::Power),
                                            ::testing::Values(2.0f)),
                         PWLApproxmiationPWLMeTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_base_pwl_me_log,
                         PWLApproxmiationPWLMeTest,
                         ::testing::Combine(::testing::ValuesIn(net_precisions),
                                            ::testing::ValuesIn(input_values_ranges_log),
                                            ::testing::Values(PWLApproximationOp::Log),
                                            ::testing::Values(0.0f)),
                         PWLApproxmiationPWLMeTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_base_pwl_me_softsign,
                         PWLApproxmiationPWLMeTest,
                         ::testing::Combine(::testing::ValuesIn(net_precisions),
                                            ::testing::ValuesIn(input_values_ranges),
                                            ::testing::Values(PWLApproximationOp::SoftSign),
                                            ::testing::Values(0.0f)),
                         PWLApproxmiationPWLMeTest::getTestCaseName);

}  // namespace LayerTestsDefinitions