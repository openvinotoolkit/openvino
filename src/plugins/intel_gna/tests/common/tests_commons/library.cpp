// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "library.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <exception>
#include <fstream>
#include <sstream>

#include "ops/pwl.hpp"

using namespace ov::opset9;
using namespace ov::intel_gna::op;

namespace tests_common {

Timer::Timer(const std::string& name,
             bool print_on_destruction,
             std::shared_ptr<std::chrono::duration<double, std::micro>> time_spent)
    : _name(name),
      _print_on_destruction(print_on_destruction),
      _time_spent(time_spent) {
    _start_time = std::chrono::steady_clock::now();
}

Timer::~Timer() {
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

double Timer::get_time_spent_in_us() {
    return get_time_spent().count();
}
std::chrono::duration<double, std::micro> Timer::get_time_spent() {
    return std::chrono::steady_clock::now() - _start_time;
}

void Timer::print_time_spent() {
    std::cout << "[" << _name << "]: " << get_time_spent_in_us() << " us" << std::endl;
    auto current = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> time_spent = current - _start_time;
}

std::ostream& operator<<(std::ostream& os, const Error& error) {
    os << "Error(" << std::setprecision(12) << " max_error(" << std::setprecision(12) << error.max_error
       << "), avg_error(" << error.avg_error() << "))";
    return os;
}

std::vector<float> count_errors(const std::vector<float>& reference_dat, const std::vector<float>& results) {
    if (reference_dat.size() != results.size()) {
        std::stringstream sstr;
        sstr << "Refernce data and results size mismatch. ";
        sstr << reference_dat.size() << " vs " << results.size();
        throw std::logic_error(sstr.str());
    }

    std::vector<float> errors(reference_dat.size(), 0.f);
    for (int i = 0; i < reference_dat.size(); i++) {
        errors[i] = std::abs(reference_dat[i] - results[i]);
    }
    return errors;
}

void ResultWriterFile::save_file(const std::string& output_file_path,
                                 const std::vector<char>& data,
                                 bool append) const {
    std::ios_base::openmode flags = std::fstream::out;
    if (append) {
        flags = flags | std::fstream::app;
    }

    std::fstream output_file(output_file_path, flags);

    output_file.write(data.data(), data.size());
    output_file.close();
}

// TODO refactor
static std::vector<char> prepare_pwl_dump_file_data_horizontally(const PWLDumpData& data) {
    const char separator = ';';
    std::stringstream output_data;

    output_data << "Function" << separator << data.get_function_name() << std::endl;
    const auto& input_data_ref = data.get_intput_data_ref();
    output_data << "Samples" << separator << input_data_ref.size() << std::endl;

    output_data << "Input data";
    for (const auto& value : input_data_ref) {
        output_data << separator << value;
    }
    output_data << std::endl;

    const auto& reference_data_ref = data.get_reference_data_ref();
    output_data << "Reference data";
    for (const auto& value : reference_data_ref) {
        output_data << separator << value;
    }
    output_data << std::endl;

    const auto& results_ref = data.get_results_ref();

    for (const auto& result : results_ref) {
        output_data << std::round(result.pwl_me) << std::endl;
        output_data << "Segments" << separator << result.segments.size() << std::endl;
        output_data << "No" << separator << "m" << separator << "b" << separator << "knot" << std::endl;
        for (size_t i = 0; i < result.segments.size(); ++i) {
            output_data << i << separator << result.segments.m[i] << separator << result.segments.b[i] << separator
                        << result.segments.knots[i] << std::endl;
        }

        output_data << "Max error" << separator << result.error_summary.max_error << std::endl;
        output_data << "Avg error" << separator << result.error_summary.avg_error() << std::endl;

        output_data << "Results";
        for (const auto& value : result.results) {
            output_data << separator << value;
        }
        output_data << std::endl;

        output_data << "Errors";
        for (const auto& value : result.errors) {
            output_data << separator << value;
        }
        output_data << std::endl;
    }

    auto output_string = output_data.str();
    return std::vector<char>(output_string.begin(), output_string.end());
}

// TODO refactor
static std::vector<char> prepare_pwl_dump_file_data_vertically(const PWLDumpData& data) {
    const char separator = ';';
    std::stringstream output_data;

    output_data << "Function" << separator << data.get_function_name() << std::endl;
    const auto& input_data_ref = data.get_intput_data_ref();
    const auto common_data_size = input_data_ref.size();
    output_data << "Samples" << separator << common_data_size << std::endl;

    // Build labels
    std::vector<std::string> main_labels;
    main_labels.push_back("NO");
    main_labels.push_back("Input data");

    const auto& reference_data_ref = data.get_reference_data_ref();
    if (common_data_size != reference_data_ref.size()) {
        std::stringstream error_message;
        error_message << "Input data and referende data size mismatch [" << common_data_size + "] vs ["
                      << reference_data_ref.size() << "]";
        throw std::logic_error(error_message.str());
    }
    main_labels.push_back("Reference data");

    const auto& results_ref = data.get_results_ref();

    size_t common_labels_size = main_labels.size();

    std::vector<std::string> segments_values(common_labels_size, std::string(""));
    segments_values[0] = "Segments";

    std::vector<std::string> max_errors(common_labels_size, std::string(""));
    max_errors[0] = "Max errors";

    std::vector<std::string> avg_errors(common_labels_size, std::string(""));
    avg_errors[0] = "Avg errors";

    for (const auto& result : results_ref) {
        if (common_data_size != result.results.size()) {
            std::stringstream error_message;
            error_message << "Input data and result data size mismatch for PWL_ME [ " << result.pwl_me << "]: ["
                          << common_data_size << "] vs[" << result.results.size() << "]";
            throw std::logic_error(error_message.str());
        }
        main_labels.push_back("PWL_ME: " + std::to_string(result.pwl_me));
        segments_values.push_back(std::to_string(result.segments.size()));
        std::stringstream max_error;
        max_error << std::setprecision(12) << result.error_summary.max_error;
        max_errors.push_back(max_error.str());
        std::stringstream avg_error;
        avg_error << std::setprecision(12) << result.error_summary.avg_error();
        avg_errors.push_back(avg_error.str());
    }

    for (const auto& result : results_ref) {
        if (result.errors.empty()) {
            continue;
        }

        if (common_data_size != result.errors.size()) {
            std::stringstream error_message;
            error_message << "Input data and error data size mismatch for PWL_ME [ " << result.pwl_me << "]: ["
                          << common_data_size << "] vs[" << result.errors.size() << "]";
            throw std::logic_error(error_message.str());
        }
        main_labels.push_back("PWL_ME: " + std::to_string(result.pwl_me) + " error");
        // segments_values.push_back(std::to_string(result.segments));
    }

    // store labels
    for (size_t i = 0; i < main_labels.size(); ++i) {
        if (i > 0) {
            output_data << separator;
        }
        output_data << main_labels[i];
    }
    output_data << std::endl;

    // store segments
    for (size_t i = 0; i < segments_values.size(); ++i) {
        if (i > 0) {
            output_data << separator;
        }
        output_data << segments_values[i];
    }
    output_data << std::endl;

    // max errors
    for (size_t i = 0; i < max_errors.size(); ++i) {
        if (i > 0) {
            output_data << separator;
        }
        output_data << max_errors[i];
    }
    output_data << std::endl;

    // avg errors
    for (size_t i = 0; i < avg_errors.size(); ++i) {
        if (i > 0) {
            output_data << separator;
        }
        output_data << avg_errors[i];
    }
    output_data << std::endl;

    for (size_t i = 0; i < common_data_size; ++i) {
        output_data << (i + 1) << separator;
        // store results
        output_data << input_data_ref[i] << separator;
        output_data << reference_data_ref[i] << separator;
        for (size_t j = 0; j < results_ref.size(); ++j) {
            if (j > 0) {
                output_data << separator;
            }
            output_data << results_ref[j].results[i];
        }

        // store errors
        for (size_t j = 0; j < results_ref.size(); ++j) {
            if (!results_ref[j].errors.empty()) {
                output_data << separator << results_ref[j].errors[i];
            }
        }
        output_data << std::endl;
    }

    auto output_string = output_data.str();
    return std::vector<char>(output_string.begin(), output_string.end());
}

std::string PWLDumpData::get_function_name() const {
    return m_function_name;
}
void PWLDumpData::set_function_name(const std::string& function_name) {
    m_function_name = function_name;
}
std::vector<float> PWLDumpData::get_intput_data() const {
    return m_input_data;
}

const std::vector<float>& PWLDumpData::get_intput_data_ref() const {
    return m_input_data;
}

void PWLDumpData::set_input_data(const std::vector<float>& data) {
    m_input_data = data;
}

std::vector<float> PWLDumpData::get_reference_data() const {
    return m_reference_data;
}

const std::vector<float>& PWLDumpData::get_reference_data_ref() const {
    return m_reference_data;
}

void PWLDumpData::set_reference_data(const std::vector<float>& data) {
    m_reference_data = data;
}

void PWLDumpData::add_results(const PWLMEResults& results) {
    m_pwl_me_results.push_back(results);
}

std::vector<PWLMEResults> PWLDumpData::get_results() const {
    return m_pwl_me_results;
}

const std::vector<PWLMEResults>& PWLDumpData::get_results_ref() const {
    return m_pwl_me_results;
}

Error ErrorValidator::CountError(const std::vector<float>& expected_buffer, const std::vector<float>& actual_buffer) {
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

bool ErrorValidator::IsPrevErrorLower(const Error& prev_error, const Error& current_error) {
    if (prev_error.max_error < current_error.max_error || prev_error.avg_error() < current_error.avg_error()) {
        return true;
    }

    return false;
}

/**
 * Return empty string if current_pwl_error is lower than previous_pwl_error.
 */

ResultCollector::ResultCollector(const ResultWriter& writer, bool count_error)
    : m_writer(writer),
      m_count_error(count_error) {}

void ResultCollector::set_function_name(const std::string& name) {
    m_dump_data.set_function_name(name);
}

void ResultCollector::set_input_data(const std::vector<float>& data) {
    m_dump_data.set_input_data(data);
}

void ResultCollector::set_reference_data(const std::vector<float>& data) {
    m_dump_data.set_reference_data(data);
}

void ResultCollector::add_results(double pwl_me,
                                  const tests_common::PWLSegments& segments,
                                  const std::vector<float>& results) {
    std::vector<float> errors;
    if (m_count_error) {
        // TODO move to error validator
        errors = count_errors(m_dump_data.get_reference_data(), results);
    }

    Error error_summary = ErrorValidator::CountError(m_dump_data.get_reference_data_ref(), results);
    m_dump_data.add_results({pwl_me, segments, results, errors, error_summary});
}

void ResultCollector::store_results(const std::string& output_file_path, const Orientation& orientation, bool append) {
    // TODO probably need to increase precsion of floating point when dumping.
    if (orientation == Orientation::VERTICAL) {
        m_writer.save_file(output_file_path, prepare_pwl_dump_file_data_vertically(m_dump_data), append);
    } else {
        m_writer.save_file(output_file_path, prepare_pwl_dump_file_data_horizontally(m_dump_data), append);
    }
}

PWLDumpData ResultCollector::get_collected_data() const {
    return m_dump_data;
}

const PWLDumpData& ResultCollector::get_collected_data_ref() const {
    return m_dump_data;
}

// TODO remove if needed
size_t get_first_pwl_segments_number(std::shared_ptr<ov::Model>& model_under_test) {
    auto ops = model_under_test->get_ordered_ops();
    for (const auto operation : ops) {
        auto pwl = std::dynamic_pointer_cast<Pwl>(operation);
        if (pwl) {
            static constexpr size_t index_of_m = 1;
            auto m_input = pwl->input_value(index_of_m);
            return ov::shape_size(m_input.get_shape());
        }
    }

    throw std::logic_error("Could not found expected layer in model!!!");
}

PWLSegments get_first_pwl_segments(std::shared_ptr<ov::Model>& model) {
    auto ops = model->get_ordered_ops();
    for (const auto operation : ops) {
        auto pwl = std::dynamic_pointer_cast<Pwl>(operation);
        if (pwl) {
            static constexpr size_t index_of_m = 1;
            static constexpr size_t index_of_b = 2;
            static constexpr size_t index_of_knots = 3;
            auto m_input = pwl->input_value(index_of_m);
            auto m_const = std::dynamic_pointer_cast<Constant>(m_input.get_node_shared_ptr());

            auto b_input = pwl->input_value(index_of_b);
            auto b_const = std::dynamic_pointer_cast<Constant>(b_input.get_node_shared_ptr());

            auto knots_input = pwl->input_value(index_of_knots);
            auto knots_const = std::dynamic_pointer_cast<Constant>(knots_input.get_node_shared_ptr());

            if (!m_const || !b_const || !knots_const) {
                // TODO unify exception types

                throw std::runtime_error("It is not possible to retrieve PWL data");
            }

            PWLSegments segments;
            segments.m = m_const->cast_vector<double>();
            segments.b = b_const->cast_vector<double>();
            segments.knots = knots_const->cast_vector<double>();

            return segments;
        }
    }

    throw std::logic_error("Could not found expected layer in model!!!");
}

//TODO remove at the end
std::vector<double> generate_pwl_me_values() {
    std::vector<double> pwl_mes;
    // for (int i = 10; i > 0; --i) {
    //    pwl_mes.push_back(i / 10.);
    //}
    //// add 1.0 to 0.1 step 0.1
    // for (int i = 10; i > 0; --i) {
    //    pwl_mes.push_back(i / 10.);
    //}

    //// add 0.09 to 0.03 step 0.01
    // for (int i = 9; i > 3; --i) {
    //    pwl_mes.push_back(i / 100.);
    //}
    // PWL_ME was replaced by number of segments.
    for (int i = 4; i <= 128; ++i) {
        pwl_mes.push_back(i);
    }
    return pwl_mes;
    // return {1.0};
}

std::vector<float> generate_input_data_covering_interal(size_t size, float lower_bound, float upper_bound) {
    if (size == 0) {
        return {};
    }

    if (size == 1) {
        return {lower_bound};
    }

    std::vector<float> data(size, 0.1);
    float step = (upper_bound - lower_bound) / (size - 1);
    float value = lower_bound;
    for (size_t i = 0; i < size; i++) {
        data[i] = value;
        value += step;
    }
    return data;
}

std::vector<float> run_reference_function(ReferenceFunction reference_function, const std::vector<float>& input_data) {
    std::vector<float> results;
    for (const auto& input : input_data) {
        results.emplace_back(reference_function(input));
    }
    return results;
}

void fail_in_case_errors_are_not_deareasing_monotonically(const tests_common::PWLDumpData& data) {
    auto& results = data.get_results_ref();
    auto prev_error = results[0].error_summary;
    for (int i = 1; i < results.size(); ++i) {
        if (tests_common::ErrorValidator::IsPrevErrorLower(prev_error, results[i].error_summary)) {
            std::stringstream error_message;
            error_message << "pwl_me[" << results[i].pwl_me << "]: " << std::setprecision(12)
                          << results[i].error_summary << ", is bigger than for previous pwl_me["
                          << std::setprecision(12) << results[i - 1].pwl_me << "]: " << prev_error;
            FAIL() << error_message.str();
        }
        prev_error = results[i].error_summary;
    }
}

}  // namespace tests_common