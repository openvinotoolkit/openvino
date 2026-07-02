// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inference.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "samples/slog.hpp"
#include "utils.hpp"

namespace {

using Time = std::chrono::steady_clock;

double get_duration_ms(const Time::time_point& start_time) {
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(Time::now() - start_time);
    return static_cast<double>(duration.count()) / 1000.0;
}

template <typename T>
using UniformDistribution = typename std::conditional<std::is_floating_point<T>::value,
                                                      std::uniform_real_distribution<T>,
                                                      std::uniform_int_distribution<T>>::type;

template <typename T, typename DistributionType = T>
void fill_tensor_random(ov::Tensor& tensor,
                        std::mt19937& generator,
                        DistributionType min_value = std::numeric_limits<uint8_t>::min(),
                        DistributionType max_value = std::numeric_limits<uint8_t>::max()) {
    auto data = tensor.data<T>();
    UniformDistribution<DistributionType> distribution(min_value, max_value);
    for (size_t index = 0; index < tensor.get_size(); index++) {
        data[index] = static_cast<T>(distribution(generator));
    }
}

template <typename T>
void fill_tensor_value(ov::Tensor& tensor, T value) {
    auto data = tensor.data<T>();
    std::fill(data, data + tensor.get_size(), value);
}

template <typename T>
void fill_integer_tensor_value_checked(ov::Tensor& tensor, int64_t value) {
    bool out_of_range = false;
    if constexpr (std::is_signed<T>::value) {
        out_of_range = value < static_cast<int64_t>(std::numeric_limits<T>::min()) ||
                       value > static_cast<int64_t>(std::numeric_limits<T>::max());
    } else {
        out_of_range = value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<T>::max();
    }

    if (out_of_range) {
        OPENVINO_THROW("Cannot fill tensor with value ",
                       value,
                       " because it exceeds element type range: ",
                       tensor.get_element_type());
    }

    fill_tensor_value<T>(tensor, static_cast<T>(value));
}

void fill_integer_tensor_value(ov::Tensor& tensor, int64_t value) {
    const auto type = tensor.get_element_type();
    if (type == ov::element::i64) {
        fill_integer_tensor_value_checked<int64_t>(tensor, value);
    } else if (type == ov::element::i32) {
        fill_integer_tensor_value_checked<int32_t>(tensor, value);
    } else if (type == ov::element::i16) {
        fill_integer_tensor_value_checked<int16_t>(tensor, value);
    } else if (type == ov::element::i8) {
        fill_integer_tensor_value_checked<int8_t>(tensor, value);
    } else if (type == ov::element::u64) {
        fill_integer_tensor_value_checked<uint64_t>(tensor, value);
    } else if (type == ov::element::u32) {
        fill_integer_tensor_value_checked<uint32_t>(tensor, value);
    } else if (type == ov::element::u16) {
        fill_integer_tensor_value_checked<uint16_t>(tensor, value);
    } else if (type == ov::element::u8) {
        fill_integer_tensor_value_checked<uint8_t>(tensor, value);
    } else if (type == ov::element::boolean) {
        if (value != 0 && value != 1) {
            OPENVINO_THROW("Cannot fill boolean tensor with value ", value, ". Expected 0 or 1.");
        }
        using BoolT = ov::fundamental_type_for<ov::element::boolean>;
        fill_tensor_value<BoolT>(tensor, static_cast<BoolT>(value));
    } else {
        OPENVINO_THROW("Cannot fill tensor with integer value. Unsupported element type: ", type);
    }
}

template <typename T>
void fill_tensor_with_indices(ov::Tensor& tensor) {
    if (tensor.get_size() != 0 && tensor.get_size() - 1 > static_cast<size_t>(std::numeric_limits<T>::max())) {
        OPENVINO_THROW("Cannot fill tensor with indices because tensor size ",
                       tensor.get_size(),
                       " exceeds element type range: ",
                       tensor.get_element_type());
    }

    auto data = tensor.data<T>();
    for (size_t index = 0; index < tensor.get_size(); ++index) {
        data[index] = static_cast<T>(index);
    }
}

void fill_integer_tensor_with_indices(ov::Tensor& tensor) {
    const auto type = tensor.get_element_type();
    if (type == ov::element::i64) {
        fill_tensor_with_indices<int64_t>(tensor);
    } else if (type == ov::element::i32) {
        fill_tensor_with_indices<int32_t>(tensor);
    } else if (type == ov::element::i16) {
        fill_tensor_with_indices<int16_t>(tensor);
    } else if (type == ov::element::i8) {
        fill_tensor_with_indices<int8_t>(tensor);
    } else if (type == ov::element::u64) {
        fill_tensor_with_indices<uint64_t>(tensor);
    } else if (type == ov::element::u32) {
        fill_tensor_with_indices<uint32_t>(tensor);
    } else if (type == ov::element::u16) {
        fill_tensor_with_indices<uint16_t>(tensor);
    } else if (type == ov::element::u8) {
        fill_tensor_with_indices<uint8_t>(tensor);
    } else {
        OPENVINO_THROW("Cannot fill tensor with indices. Unsupported element type: ", type);
    }
}

size_t get_current_sequence_length(const ov::CompiledModel& compiled_model) {
    for (const auto& input : compiled_model.inputs()) {
        const auto shape = input.get_partial_shape();
        if (!shape.is_static() || shape.size() != 3) {
            continue;
        }

        const auto name = input.get_any_name();
        if (contains_substring(name, "embed") || contains_substring(name, "hidden")) {
            return shape[1].get_length();
        }
    }

    return 1;
}

size_t get_batch_size(const ov::CompiledModel& compiled_model) {
    for (const auto& input : compiled_model.inputs()) {
        const auto shape = input.get_partial_shape();
        if (shape.is_static() && shape.size() > 0) {
            return shape[0].get_length();
        }
    }

    return 1;
}

struct LatencyStatistics {
    double median = 0.0;
    double average = 0.0;
    double min = 0.0;
    double max = 0.0;
    bool median_is_approximate = false;
};

class LatencyStatisticsCollector {
public:
    explicit LatencyStatisticsCollector(size_t iterations) {
        constexpr size_t max_median_samples = 10000;
        m_sample_step = iterations <= max_median_samples ? 1 : iterations / max_median_samples;
        if (iterations > max_median_samples && iterations % max_median_samples != 0) {
            ++m_sample_step;
        }

        m_median_samples.reserve(std::min(iterations, max_median_samples));
    }

    void add(double latency) {
        if (m_count == 0) {
            m_min = latency;
            m_max = latency;
        } else {
            m_min = std::min(m_min, latency);
            m_max = std::max(m_max, latency);
        }

        m_sum += latency;
        if (m_count % m_sample_step == 0) {
            m_median_samples.push_back(latency);
        }
        ++m_count;
    }

    LatencyStatistics get() {
        if (m_count == 0) {
            return {};
        }

        std::sort(m_median_samples.begin(), m_median_samples.end());
        const auto middle = m_median_samples.size() / 2;
        const auto median = m_median_samples.size() % 2 == 0
                                ? (m_median_samples[middle - 1] + m_median_samples[middle]) / 2.0
                                : m_median_samples[middle];

        return {median, m_sum / static_cast<double>(m_count), m_min, m_max, m_median_samples.size() != m_count};
    }

private:
    size_t m_sample_step = 1;
    size_t m_count = 0;
    double m_sum = 0.0;
    double m_min = 0.0;
    double m_max = 0.0;
    std::vector<double> m_median_samples;
};

void fill_tensor_with_random_data(ov::Tensor& tensor, std::mt19937& generator) {
    const auto type = tensor.get_element_type();
    if (type == ov::element::f32) {
        fill_tensor_random<float, float>(tensor, generator);
    } else if (type == ov::element::f64) {
        fill_tensor_random<double, double>(tensor, generator);
    } else if (type == ov::element::f16) {
        fill_tensor_random<ov::float16, float>(tensor, generator);
    } else if (type == ov::element::bf16) {
        fill_tensor_random<ov::bfloat16, float>(tensor, generator);
    } else if (type == ov::element::i64) {
        fill_tensor_random<int64_t, int64_t>(tensor, generator);
    } else if (type == ov::element::i32) {
        fill_tensor_random<int32_t, int32_t>(tensor, generator);
    } else if (type == ov::element::i16) {
        fill_tensor_random<int16_t, int16_t>(tensor, generator);
    } else if (type == ov::element::i8) {
        fill_tensor_random<int8_t, int32_t>(tensor,
                                            generator,
                                            std::numeric_limits<int8_t>::min(),
                                            std::numeric_limits<int8_t>::max());
    } else if (type == ov::element::u64) {
        fill_tensor_random<uint64_t, uint64_t>(tensor, generator);
    } else if (type == ov::element::u32) {
        fill_tensor_random<uint32_t, uint32_t>(tensor, generator);
    } else if (type == ov::element::u16) {
        fill_tensor_random<uint16_t, uint16_t>(tensor, generator);
    } else if (type == ov::element::u8) {
        fill_tensor_random<uint8_t, uint32_t>(tensor, generator);
    } else if (type == ov::element::boolean) {
        fill_tensor_random<ov::fundamental_type_for<ov::element::boolean>, uint32_t>(tensor, generator, 0, 1);
    } else if (tensor.get_byte_size() != 0) {
        std::memset(tensor.data(), 0, tensor.get_byte_size());
    }
}

ov::Shape get_inference_tensor_shape(const ov::Output<const ov::Node>& input,
                                     const std::map<std::string, ov::PartialShape>& data_shapes) {
    const auto partial_shape = input.get_partial_shape();
    auto data_shape = data_shapes.find(input.get_any_name());
    if (data_shape == data_shapes.end()) {
        data_shape = data_shapes.find(input.get_node_shared_ptr()->get_friendly_name());
    }

    if (data_shape != data_shapes.end()) {
        if (data_shape->second.is_dynamic()) {
            OPENVINO_THROW("Data shape for input '", input.get_any_name(), "' must be static: ", data_shape->second);
        }
        if (!partial_shape.compatible(data_shape->second)) {
            OPENVINO_THROW("Data shape ",
                           data_shape->second,
                           " is not compatible with model input '",
                           input.get_any_name(),
                           "' shape ",
                           partial_shape);
        }
        return data_shape->second.to_shape();
    }

    if (partial_shape.is_dynamic()) {
        OPENVINO_THROW("Input '",
                       input.get_any_name(),
                       "' has dynamic shape ",
                       partial_shape,
                       ". Please provide concrete dimensions using -shape or -data_shape before running inference.");
    }

    return partial_shape.to_shape();
}

ov::Tensor create_input_tensor(const ov::Output<const ov::Node>& input,
                               size_t current_sequence_length,
                               const std::map<std::string, ov::PartialShape>& data_shapes,
                               std::mt19937& generator) {
    const auto tensor_shape = get_inference_tensor_shape(input, data_shapes);

    if (input.get_element_type() == ov::element::string) {
        OPENVINO_THROW("String input '", input.get_any_name(), "' is not supported by hello_affinity inference mode.");
    }

    const auto name = input.get_any_name();
    ov::Tensor tensor(input.get_element_type(), tensor_shape);
    if (contains_substring(name, "beam_idx")) {
        fill_integer_tensor_with_indices(tensor);
    } else if (contains_substring(name, "past_seq_len")) {
        fill_integer_tensor_value(tensor, 0);
    } else if (contains_substring(name, "total_seq_len")) {
        fill_integer_tensor_value(tensor, static_cast<int64_t>(current_sequence_length));
    } else {
        fill_tensor_with_random_data(tensor, generator);
    }

    return tensor;
}

}  // namespace

double run_inference(ov::CompiledModel& compiled_model,
                     const std::string& data_shape_string,
                     size_t iterations,
                     bool skip_warmup) {
    if (iterations == 0) {
        OPENVINO_THROW("Number of inference iterations must be greater than zero.");
    }

    ov::InferRequest infer_request = compiled_model.create_infer_request();
    const auto data_shapes = data_shape_string.empty() ? std::map<std::string, ov::PartialShape>{}
                                                       : parse_input_shapes(data_shape_string, compiled_model.inputs());
    const auto current_sequence_length = get_current_sequence_length(compiled_model);
    std::mt19937 generator(0);
    slog::info << "Current sequence length hint: " << current_sequence_length << slog::endl;
    for (const auto& input : compiled_model.inputs()) {
        auto tensor = create_input_tensor(input, current_sequence_length, data_shapes, generator);
        slog::info << "Setting input tensor " << input.get_any_name() << " : " << tensor.get_element_type() << " / "
                   << tensor.get_shape() << slog::endl;
        infer_request.set_tensor(input.get_any_name(), tensor);
    }

    if (!skip_warmup) {
        slog::info << "Starting warm-up inference" << slog::endl;
        const auto warmup_start_time = Time::now();
        infer_request.infer();
        const auto warmup_time_ms = get_duration_ms(warmup_start_time);
        slog::info << "First inference took " << format_duration_ms(warmup_time_ms) << " ms" << slog::endl;
    } else {
        slog::info << "Skipping warm-up inference due to -no_warmup flag" << slog::endl;
    }

    slog::info << "Starting inference" << slog::endl;
    slog::info << "Inference iterations: " << iterations << slog::endl;
    LatencyStatisticsCollector latency_statistics_collector(iterations);

    const auto infer_start_time = Time::now();
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        const auto iteration_start_time = Time::now();
        infer_request.infer();
        latency_statistics_collector.add(get_duration_ms(iteration_start_time));
    }
    const auto total_infer_time_ms = get_duration_ms(infer_start_time);
    const auto batch_size = get_batch_size(compiled_model);
    const auto throughput = total_infer_time_ms == 0.0 ? 0.0 : 1000.0 * batch_size * iterations / total_infer_time_ms;
    const auto latency_statistics = latency_statistics_collector.get();

    slog::info << "Inference completed successfully" << slog::endl;

    try {
        const auto execution_devices = compiled_model.get_property(ov::execution_devices);
        slog::info << "Execution Devices: " << execution_devices << slog::endl;
    } catch (const ov::Exception&) {
    }

    slog::info << "Count:               " << iterations << " iterations" << slog::endl;
    slog::info << "Duration:            " << format_duration_ms(total_infer_time_ms) << " ms" << slog::endl;
    slog::info << "Latency:" << slog::endl;
    slog::info << "   Median" << (latency_statistics.median_is_approximate ? " (approx):  " : ":           ")
               << format_duration_ms(latency_statistics.median) << " ms" << slog::endl;
    slog::info << "   Average:          " << format_duration_ms(latency_statistics.average) << " ms" << slog::endl;
    slog::info << "   Min:              " << format_duration_ms(latency_statistics.min) << " ms" << slog::endl;
    slog::info << "   Max:              " << format_duration_ms(latency_statistics.max) << " ms" << slog::endl;
    slog::info << "Throughput:          " << format_double(throughput) << " FPS" << slog::endl;
    return total_infer_time_ms;
}
