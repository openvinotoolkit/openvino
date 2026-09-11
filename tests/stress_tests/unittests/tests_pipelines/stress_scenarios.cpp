// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stress_scenarios.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

class ThreadErrors {
public:
    void add(const std::string& context, const std::string& message) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_errors.emplace_back(context + ": " + message);
    }

    template <typename Callable>
    void run(const std::string& context, Callable&& callable) {
        try {
            callable();
        } catch (const std::exception& error) {
            add(context, error.what());
        } catch (...) {
            add(context, "unknown exception");
        }
    }

    void throw_if_any() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_errors.empty()) {
            return;
        }

        std::ostringstream message;
        message << m_errors.size() << " worker failure(s)";
        for (const auto& error : m_errors) {
            message << '\n' << error;
        }
        throw std::runtime_error(message.str());
    }

private:
    mutable std::mutex m_mutex;
    std::vector<std::string> m_errors;
};

template <typename Callable>
void run_workers(int thread_count, Callable&& callable) {
    ThreadErrors errors;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(thread_count));
    for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
        workers.emplace_back([&, thread_index]() {
            errors.run("worker " + std::to_string(thread_index), [&]() {
                callable(thread_index);
            });
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    errors.throw_if_any();
}

int normalized_threads(int threads) {
    return std::max(1, threads);
}

int normalized_iterations(int iterations) {
    return std::max(1, iterations);
}

size_t resolve_dimension(const ov::Dimension& dimension, size_t preferred) {
    if (dimension.is_static()) {
        return static_cast<size_t>(dimension.get_length());
    }

    size_t value = preferred;
    if (dimension.get_min_length() != -1) {
        value = std::max(value, static_cast<size_t>(dimension.get_min_length()));
    }
    if (dimension.get_max_length() != -1) {
        value = std::min(value, static_cast<size_t>(dimension.get_max_length()));
    }
    return value;
}

ov::Shape resolve_shape(const ov::PartialShape& partial_shape, size_t sequence_length) {
    if (partial_shape.rank().is_dynamic()) {
        throw std::runtime_error("Cannot create an input tensor for a model with dynamic rank");
    }

    ov::Shape shape;
    shape.reserve(partial_shape.size());
    for (size_t index = 0; index < partial_shape.size(); ++index) {
        shape.push_back(resolve_dimension(partial_shape[index], index == 0 ? 1 : sequence_length));
    }
    return shape;
}

template <typename ElementType>
void fill_tensor(ov::Tensor& tensor, ElementType value) {
    std::fill_n(tensor.data<ElementType>(), tensor.get_size(), value);
}

void fill_tensor(ov::Tensor& tensor, const std::string& input_name, size_t sequence_length) {
    const bool is_attention_mask = input_name.find("attention_mask") != std::string::npos;
    const bool is_position_ids = input_name.find("position_ids") != std::string::npos;
    const bool is_input_ids = input_name.find("input_ids") != std::string::npos;

    if (tensor.get_element_type() == ov::element::string) {
        fill_tensor(tensor, std::string{});
    } else if (tensor.get_element_type() == ov::element::boolean) {
        fill_tensor(tensor, is_attention_mask || is_input_ids || is_position_ids);
    } else if (tensor.get_element_type() == ov::element::i8) {
        fill_tensor(tensor, static_cast<int8_t>(is_attention_mask || is_input_ids || is_position_ids));
    } else if (tensor.get_element_type() == ov::element::u8) {
        fill_tensor(tensor, static_cast<uint8_t>(is_attention_mask || is_input_ids || is_position_ids));
    } else if (tensor.get_element_type() == ov::element::i16) {
        fill_tensor(tensor, static_cast<int16_t>(is_attention_mask || is_input_ids || is_position_ids));
    } else if (tensor.get_element_type() == ov::element::u16) {
        fill_tensor(tensor, static_cast<uint16_t>(is_attention_mask || is_input_ids || is_position_ids));
    } else if (tensor.get_element_type() == ov::element::i32) {
        fill_tensor(tensor, static_cast<int32_t>(is_attention_mask || is_input_ids || is_position_ids));
    } else if (tensor.get_element_type() == ov::element::u32) {
        fill_tensor(tensor, static_cast<uint32_t>(is_attention_mask || is_input_ids || is_position_ids));
    } else if (tensor.get_element_type() == ov::element::i64) {
        fill_tensor(tensor, static_cast<int64_t>(is_attention_mask || is_input_ids || is_position_ids));
    } else if (tensor.get_element_type() == ov::element::u64) {
        fill_tensor(tensor, static_cast<uint64_t>(is_attention_mask || is_input_ids || is_position_ids));
    } else if (tensor.get_element_type() == ov::element::f16) {
        fill_tensor(tensor, ov::float16{0.f});
    } else if (tensor.get_element_type() == ov::element::bf16) {
        fill_tensor(tensor, ov::bfloat16{0.f});
    } else if (tensor.get_element_type() == ov::element::f32) {
        fill_tensor(tensor, 0.f);
    } else if (tensor.get_element_type() == ov::element::f64) {
        fill_tensor(tensor, 0.0);
    } else {
        std::memset(tensor.data(), 0, tensor.get_byte_size());
    }

    if (is_position_ids && tensor.get_size() > 1 && tensor.get_element_type() != ov::element::string) {
        const size_t position_count = std::min(sequence_length, tensor.get_size());
        if (tensor.get_element_type() == ov::element::i64) {
            auto* data = tensor.data<int64_t>();
            for (size_t index = 0; index < position_count; ++index) {
                data[index] = static_cast<int64_t>(index);
            }
        } else if (tensor.get_element_type() == ov::element::i32) {
            auto* data = tensor.data<int32_t>();
            for (size_t index = 0; index < position_count; ++index) {
                data[index] = static_cast<int32_t>(index);
            }
        }
    }
}

ov::Tensor make_tensor(const ov::Output<const ov::Node>& port,
                       const std::string& input_name,
                       size_t sequence_length = 1) {
    ov::Tensor tensor(port.get_element_type(), resolve_shape(port.get_partial_shape(), sequence_length));
    fill_tensor(tensor, input_name, sequence_length);
    return tensor;
}

void set_inputs(ov::InferRequest& request,
                const ov::CompiledModel& compiled_model,
                size_t sequence_length = 1) {
    for (const auto& input : compiled_model.inputs()) {
        request.set_tensor(input, make_tensor(input, input.get_any_name(), sequence_length));
    }
}

bool is_language_model(const ov::CompiledModel& compiled_model) {
    for (const auto& input : compiled_model.inputs()) {
        const auto& name = input.get_any_name();
        if (name.find("input_ids") != std::string::npos || name.find("attention_mask") != std::string::npos ||
            name.find("position_ids") != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool supports_export_import(ov::Core& core, const std::string& device) {
    const auto capabilities = core.get_property(device, ov::device::capabilities);
    return std::find(capabilities.begin(), capabilities.end(), ov::device::capability::EXPORT_IMPORT) !=
           capabilities.end();
}

}  // namespace

void stress_load_unload(const std::string& model,
                        const std::string& device,
                        int iterations,
                        int threads) {
    ov::Core core;
    const auto network = core.read_model(model);
    run_workers(normalized_threads(threads), [&](int) {
        for (int iteration = 0; iteration < normalized_iterations(iterations); ++iteration) {
            auto compiled_model = core.compile_model(network, device);
            auto request = compiled_model.create_infer_request();
            set_inputs(request, compiled_model);
            request.infer();
            static_cast<void>(request.get_output_tensor(0));
        }
    });
}

void stress_parallel_infer(const std::string& model,
                           const std::string& device,
                           int iterations,
                           int threads) {
    ov::Core core;
    auto compiled_model = core.compile_model(model, device);
    const bool language_model = is_language_model(compiled_model);
    run_workers(normalized_threads(threads), [&](int thread_index) {
        auto request = compiled_model.create_infer_request();
        for (int iteration = 0; iteration < normalized_iterations(iterations); ++iteration) {
            const size_t sequence_length =
                language_model ? static_cast<size_t>(8 + ((iteration + thread_index) % 16)) : 1;
            set_inputs(request, compiled_model, sequence_length);
            request.infer();
            static_cast<void>(request.get_output_tensor(0));
        }
    });
}

void stress_concurrent_load_infer(const std::string& model,
                                  const std::string& device,
                                  int iterations,
                                  int threads) {
    ov::Core core;
    const auto network = core.read_model(model);
    auto stable_model = core.compile_model(network, device);
    const int worker_count = normalized_threads(threads);
    const int load_workers = std::max(1, worker_count / 2);
    const int infer_workers = std::max(1, worker_count - load_workers);
    std::atomic<bool> start{false};
    ThreadErrors errors;
    std::vector<std::thread> workers;

    for (int index = 0; index < load_workers; ++index) {
        workers.emplace_back([&, index]() {
            errors.run("load worker " + std::to_string(index), [&]() {
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                for (int iteration = 0; iteration < normalized_iterations(iterations); ++iteration) {
                    auto temporary_model = core.compile_model(network, device);
                    static_cast<void>(temporary_model);
                }
            });
        });
    }
    for (int index = 0; index < infer_workers; ++index) {
        workers.emplace_back([&, index]() {
            errors.run("infer worker " + std::to_string(index), [&]() {
                auto request = stable_model.create_infer_request();
                set_inputs(request, stable_model);
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                for (int iteration = 0; iteration < normalized_iterations(iterations); ++iteration) {
                    request.infer();
                    static_cast<void>(request.get_output_tensor(0));
                }
            });
        });
    }

    start.store(true, std::memory_order_release);
    for (auto& worker : workers) {
        worker.join();
    }
    errors.throw_if_any();
}

void stress_import_export(const std::string& model,
                          const std::string& device,
                          int iterations,
                          int threads) {
    ov::Core core;
    if (!supports_export_import(core, device)) {
        return;
    }

    auto compiled_model = core.compile_model(model, device);
    std::ostringstream output;
    compiled_model.export_model(output);
    const std::string blob = output.str();

    run_workers(normalized_threads(threads), [&](int) {
        for (int iteration = 0; iteration < normalized_iterations(iterations); ++iteration) {
            std::istringstream input(blob);
            auto imported_model = core.import_model(input, device);
            auto request = imported_model.create_infer_request();
            set_inputs(request, imported_model);
            request.infer();
            static_cast<void>(request.get_output_tensor(0));
        }
    });
}

void stress_mid_flight_cancel(const std::string& model,
                              const std::string& device,
                              int iterations,
                              int threads) {
    ov::Core core;
    auto compiled_model = core.compile_model(model, device);
    const int request_count = std::max(2, normalized_threads(threads));

    for (int iteration = 0; iteration < normalized_iterations(iterations); ++iteration) {
        std::vector<ov::InferRequest> requests;
        requests.reserve(static_cast<size_t>(request_count));
        for (int index = 0; index < request_count; ++index) {
            requests.emplace_back(compiled_model.create_infer_request());
            set_inputs(requests.back(), compiled_model);
        }

        ThreadErrors errors;
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(request_count));
        std::atomic<bool> cancel_requested{false};
        for (int index = 0; index < request_count; ++index) {
            workers.emplace_back([&, index]() {
                try {
                    requests[static_cast<size_t>(index)].infer();
                    static_cast<void>(requests[static_cast<size_t>(index)].get_output_tensor(0));
                } catch (const std::exception& error) {
                    if (index != 0 || !cancel_requested.load(std::memory_order_acquire)) {
                        errors.add("infer worker " + std::to_string(index), error.what());
                    }
                } catch (...) {
                    errors.add("infer worker " + std::to_string(index), "unknown exception");
                }
            });
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        cancel_requested.store(true, std::memory_order_release);
        errors.run("cancel request", [&]() {
            requests.front().cancel();
        });
        for (auto& worker : workers) {
            worker.join();
        }
        errors.throw_if_any();
    }
}

void stress_memory_pressure(const std::string& model,
                            const std::string& device,
                            int iterations,
                            int threads) {
    ov::Core core;
    const auto network = core.read_model(model);
    const int model_count = normalized_threads(threads);
    for (int iteration = 0; iteration < normalized_iterations(iterations); ++iteration) {
        std::vector<ov::CompiledModel> compiled_models;
        compiled_models.reserve(static_cast<size_t>(model_count));
        for (int index = 0; index < model_count; ++index) {
            compiled_models.emplace_back(core.compile_model(network, device));
        }
        run_workers(model_count, [&](int index) {
            auto& compiled_model = compiled_models[static_cast<size_t>(index)];
            auto request = compiled_model.create_infer_request();
            set_inputs(request, compiled_model);
            request.infer();
            static_cast<void>(request.get_output_tensor(0));
        });
    }
}

void stress_destroy_compiled_model(const std::string& model,
                                   const std::string& device,
                                   int iterations,
                                   int threads) {
    ov::Core core;
    const auto network = core.read_model(model);
    run_workers(normalized_threads(threads), [&](int) {
        for (int iteration = 0; iteration < normalized_iterations(iterations); ++iteration) {
            auto compiled_model = std::make_shared<ov::CompiledModel>(
                core.compile_model(network, device));
            auto request = compiled_model->create_infer_request();
            set_inputs(request, *compiled_model);
            ThreadErrors errors;
            std::thread infer_thread([&]() {
                errors.run("inference", [&]() {
                    request.infer();
                    static_cast<void>(request.get_output_tensor(0));
                });
            });
            compiled_model.reset();
            infer_thread.join();
            errors.throw_if_any();
        }
    });
}

void stress_multiple_cores(const std::string& model,
                           const std::string& device,
                           int iterations,
                           int threads) {
    run_workers(normalized_threads(threads), [&](int) {
        for (int iteration = 0; iteration < normalized_iterations(iterations); ++iteration) {
            ov::Core core;
            auto compiled_model = core.compile_model(model, device);
            auto request = compiled_model.create_infer_request();
            set_inputs(request, compiled_model);
            request.infer();
            static_cast<void>(request.get_output_tensor(0));
        }
    });
}

void run_stress_scenario(const std::string& scenario,
                         const std::string& model,
                         const std::string& device,
                         int iterations,
                         int threads) {
    if (scenario == "stress_load_unload") {
        stress_load_unload(model, device, iterations, threads);
    } else if (scenario == "stress_parallel_infer") {
        stress_parallel_infer(model, device, iterations, threads);
    } else if (scenario == "stress_concurrent_load_infer") {
        stress_concurrent_load_infer(model, device, iterations, threads);
    } else if (scenario == "stress_import_export") {
        stress_import_export(model, device, iterations, threads);
    } else if (scenario == "stress_mid_flight_cancel") {
        stress_mid_flight_cancel(model, device, iterations, threads);
    } else if (scenario == "stress_memory_pressure") {
        stress_memory_pressure(model, device, iterations, threads);
    } else if (scenario == "stress_destroy_compiled_model") {
        stress_destroy_compiled_model(model, device, iterations, threads);
    } else if (scenario == "stress_multiple_cores") {
        stress_multiple_cores(model, device, iterations, threads);
    } else {
        throw std::invalid_argument("Unknown stress scenario: " + scenario);
    }
}