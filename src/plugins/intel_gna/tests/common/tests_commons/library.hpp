// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iomanip>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/opsets/opset9.hpp"
// TODO refactor and split

namespace tests_common {

struct Timer {
public:
    Timer(const std::string& name,
          bool print_on_destruction = false,
          std::shared_ptr<std::chrono::duration<double, std::micro>> time_spent = nullptr);
    ~Timer();

    double get_time_spent_in_us();
    std::chrono::duration<double, std::micro> get_time_spent();
    void print_time_spent();

private:
    std::string _name;
    std::chrono::time_point<std::chrono::steady_clock> _start_time;
    std::shared_ptr<std::chrono::duration<double, std::micro>> _time_spent;
    bool _print_on_destruction;
};

using ReferenceFunction = std::function<float(float)>;

template <typename T>
struct ReferenceFunctionFactory {};

template <>
struct ReferenceFunctionFactory<ov::opset9::Sigmoid> {
    static ReferenceFunction get_function() {
        return [](const float x) {
            return static_cast<float>(0.5 * (1.0 + std::tanh(static_cast<double>(x) / 2.0)));
            // return static_cast<float>(1.0 / (1.0 + exp(static_cast<double>(-x))));
        };
    }
};

template <>
struct ReferenceFunctionFactory<ov::opset9::Tanh> {
    static ReferenceFunction get_function() {
        return [](const float x) {
            return static_cast<float>(std::tanh(static_cast<double>(x)));
        };
    }
};

template <>
struct ReferenceFunctionFactory<ov::opset9::SoftSign> {
    static ReferenceFunction get_function() {
        return [](const double x) {
            return x / (1.0 + std::abs(x));
        };
    }
};

struct PWLSegments {
    std::vector<double> m;
    std::vector<double> b;
    std::vector<double> knots;

    size_t size() const {
        return m.size();
    }
};

// TODO move to library
PWLSegments get_first_pwl_segments(std::shared_ptr<ov::Model>& model);
size_t get_first_pwl_segments_number(std::shared_ptr<ov::Model>& model_under_test);

struct Error {
    double max_error;
    double sum_of_errors;
    size_t num_of_errors;
    // TODO this has to be unified
    double avg_error() const {
        if (num_of_errors == 0) {
            return 0;
        }
        return sum_of_errors / num_of_errors;
    }
};
std::ostream& operator<<(std::ostream& os, const Error& error);

class ErrorValidator {
public:
    static Error CountError(const std::vector<float>& expected_buffer, const std::vector<float>& actual_buffer);

    static bool IsPrevErrorLower(const Error& prev_error, const Error& current_error);
};

class ResultWriter {
public:
    virtual ~ResultWriter() = default;
    virtual void save_file(const std::string& output_file_path, const std::vector<char>& data, bool append) const = 0;
};

class ResultWriterFile : public ResultWriter {
public:
    void save_file(const std::string& output_file_path, const std::vector<char>& data, bool append) const override;
};

struct PWLMEResults {
    double pwl_me = 0.0;
    PWLSegments segments;
    std::vector<float> results;
    std::vector<float> errors;
    Error error_summary;
};

enum class Orientation { VERTICAL, HORIOZONTAL };

class PWLDumpData /* : public DataToFileConverter */ {
public:
    PWLDumpData() = default;

    std::string get_function_name() const;
    void set_function_name(const std::string& function_name);
    const std::vector<float>& get_intput_data_ref() const;
    std::vector<float> get_intput_data() const;
    void set_input_data(const std::vector<float>& data);

    std::vector<float> get_reference_data() const;
    const std::vector<float>& get_reference_data_ref() const;
    void set_reference_data(const std::vector<float>& data);
    void add_results(const PWLMEResults& results);
    std::vector<PWLMEResults> get_results() const;
    const std::vector<PWLMEResults>& get_results_ref() const;

private:
    std::string m_function_name;
    std::vector<float> m_input_data;
    std::vector<float> m_reference_data;
    std::vector<PWLMEResults> m_pwl_me_results;
};

class ResultCollector {
public:
    ResultCollector(const ResultWriter& writer, bool count_error = false);
    void set_function_name(const std::string& name);
    void set_input_data(const std::vector<float>& data);
    void set_reference_data(const std::vector<float>& data);
    void add_results(double pwl_me, const tests_common::PWLSegments& segments, const std::vector<float>& results);
    void store_results(const std::string& output_file_path,
                       const Orientation& orientation = Orientation::HORIOZONTAL,
                       bool append = false);
    PWLDumpData get_collected_data() const;
    const PWLDumpData& get_collected_data_ref() const;

private:
    const ResultWriter& m_writer;
    bool m_count_error;
    PWLDumpData m_dump_data;
};

/**
 * Generate PWLs.
 */
std::vector<double> generate_pwl_me_values();

/**
 * Generates inputs from lower_bound to upper_bound with step dependent on shape.
 */
std::vector<float> generate_input_data_covering_interal(size_t size, float lower_bound, float upper_bound);

std::vector<float> run_reference_function(ReferenceFunction reference_function, const std::vector<float>& input_data);

void fail_in_case_errors_are_not_deareasing_monotonically(const tests_common::PWLDumpData& data);

template <typename T>
std::shared_ptr<ov::Model> create_model(ov::element::Type precision,
                                        const ov::Shape& input_shape,
                                        const std::string model_name) {
    auto input_params = std::make_shared<ov::opset9::Parameter>(precision, input_shape);
    auto activation = std::make_shared<T>(input_params);
    activation->set_friendly_name("Activation");
    auto result = std::make_shared<ov::opset9::Result>(activation);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_params}, model_name);
}

}  // namespace tests_common
