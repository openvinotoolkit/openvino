// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdint>
#include <ctime>
#include <memory>
#include <random>

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/utils/philox_converter.hpp"
#include "openvino/reference/utils/philox_generator.hpp"

namespace ov {
namespace reference {

// TensorFlow RandomPoisson: Skip(kReserveSamplesPerExecution * output_index) before uniforms per element.
static constexpr size_t kReserveSamplesPerExecution = 256;

namespace {

// Same carry rule as TensorflowPhiloxGenerator::get_next_state (logical skip on exported state pair).
inline std::pair<uint64_t, uint64_t> tensorflow_philox_skip(std::pair<uint64_t, uint64_t> previous_state,
                                                            uint64_t skip_count) {
    const uint64_t new_n = previous_state.first + skip_count;
    const uint64_t new_op_seed = previous_state.second + (new_n < skip_count ? 1UL : 0UL);
    return {new_n, new_op_seed};
}

}  // namespace

// Uniform Sampler to keep track of the state of the uniform distribution
// Such that no sampled elements are lost
// Pytorch consume forward
// TensorFlow consume backward
struct UniformSampler {
    double* temp_output_buffer;
    size_t available_elements = 0;
    size_t cursor_index = 0;
    op::PhiloxAlignment alignment;
    std::shared_ptr<philox::PhiloxGenerator> generator;
    std::shared_ptr<philox::PhiloxConverter> converter;
    size_t* philox_random_invocations = nullptr;

    UniformSampler(double* temp_output_buffer,
                   std::shared_ptr<philox::PhiloxGenerator> generator,
                   std::shared_ptr<philox::PhiloxConverter> converter,
                   op::PhiloxAlignment alignment,
                   size_t* philox_random_invocations = nullptr)
        : temp_output_buffer(temp_output_buffer),
          generator(std::move(generator)),
          converter(std::move(converter)),
          alignment(alignment),
          philox_random_invocations(philox_random_invocations) {}

    double next() {
        if (alignment == op::PhiloxAlignment::PYTORCH) {
            if (cursor_index >= available_elements) {
                const auto& result = generator->random();
                if (philox_random_invocations != nullptr) {
                    ++(*philox_random_invocations);
                }
                converter->convert(result, 0);
                available_elements = converter->get_converted_elements_count();
                cursor_index = 0;
            }
            return temp_output_buffer[cursor_index++];
        }
        if (cursor_index == 0) {
            const auto& result = generator->random();
            if (philox_random_invocations != nullptr) {
                ++(*philox_random_invocations);
            }
            converter->convert(result, 0);
            available_elements = converter->get_converted_elements_count();
            cursor_index = available_elements;
        }
        return temp_output_buffer[--cursor_index];
    }
};

// Knuth's algorithm for generating Poisson distributed random numbers Donald E. Knuth (1969)
template <typename T>
T knuth_poisson(T lambda, UniformSampler& uniform_sampler) {
    auto enlam = std::exp(-lambda);
    T X = 0;
    auto prod = 1.0;
    while (true) {
        auto U_i = uniform_sampler.next();
        prod *= U_i;
        if (prod > enlam) {
            X += 1;
        } else {
            return static_cast<T>(X);
        }
    }
    return static_cast<T>(X);
}

// Transformed rejection method for generating Poisson distributed random numbers (Hoermann, 1993)
template <typename T>
T transformed_rejection_method_hoermann(T lambda, UniformSampler& uniform_sampler) {
    double slam = std::sqrt(lambda);
    double loglam = std::log(lambda);
    double b = 0.931 + 2.53 * slam;
    double a = -0.059 + 0.02483 * b;
    double invalpha = 1.1239 + 1.1328 / (b - 3.4);
    double vr = 0.9277 - 3.6224 / (b - 2);

    while (true) {
        double U_i = uniform_sampler.next();
        double V_i = uniform_sampler.next();
        auto u = U_i - 0.5;
        auto us = 0.5 - std::fabs(u);
        auto k = std::floor((2 * a / us + b) * u + lambda + 0.43);
        if ((us >= 0.07) && (V_i <= vr)) {
            return static_cast<T>(k);
        }
        if ((k < 0) || ((us < 0.013) && (V_i > us))) {
            continue;
        }
        if ((std::log(V_i) + std::log(invalpha) - std::log(a / (us * us) + b)) <=
            (-lambda + k * loglam - std::lgamma(k + 1))) {
            return static_cast<T>(k);
        }
    }
    return static_cast<T>(0);
}

template <typename T>
std::pair<uint64_t, uint64_t> random_poisson(const T* input_tensor,  // rates tensor aka lambda values
                                                                     // const T* generator_tensor,
                                             T* output_tensor,
                                             const Shape& input_shape,
                                             // const Shape& generator_shape,
                                             const Shape& output_shape,
                                             uint64_t seed,
                                             uint64_t seed2,
                                             std::pair<uint64_t, uint64_t> prev_state,
                                             ov::op::PhiloxAlignment alignment) {
    // When both seed values are equal to zero RandomPoisson should generate non-deterministic sequence.
    // Implementation in plugins may differ for this case.
    if (seed == 0 && seed2 == 0) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        std::mt19937_64 gen(static_cast<uint64_t>(std::time(nullptr)));
        seed = gen();
    }

    // Calculate total element count for input and output shapes
    size_t input_element_count = shape_size(input_shape);
    size_t output_element_count = shape_size(output_shape);

    if (output_element_count != input_element_count) {
        OPENVINO_THROW("RandomPoisson: output shape and input shape must have the same number of elements");
    }

    // Temporary buffer for the uniform distribution, so the philox generator can be used to generate random numbers
    // and the philox converter can be used to convert the random numbers to the target type
    std::vector<double> temp_output_buffer_vector(philox::ELEMENTS_PER_EXECUTION);
    char* temp_output_buffer_ptr = reinterpret_cast<char*>(temp_output_buffer_vector.data());
    // min and max values for the uniform distribution
    double min_val = 0.0;
    double max_val = 1.0;
    const char* min_val_ptr = reinterpret_cast<const char*>(&min_val);
    const char* max_val_ptr = reinterpret_cast<const char*>(&max_val);
    std::shared_ptr<philox::PhiloxConverter> converter = philox::make_philox_converter(temp_output_buffer_ptr,
                                                                                       element::f64,
                                                                                       philox::ELEMENTS_PER_EXECUTION,
                                                                                       min_val_ptr,
                                                                                       max_val_ptr,
                                                                                       alignment);

    if (alignment == op::PhiloxAlignment::PYTORCH) {
        std::shared_ptr<philox::PhiloxGenerator> generator =
            philox::make_philox_generator(seed, seed2, prev_state, output_element_count, alignment);
        UniformSampler uniform_sampler(temp_output_buffer_vector.data(), generator, converter, alignment);
        // For each element in the input tensor, generate a random poisson distributed random number
        // https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/random-poisson-v2#:~:text=This%20op%20uses,2.%20Addison%20Wesley
        for (size_t i = 0; i < output_element_count; i++) {
            // if lambda < 0, throw an exception, rates cannot be negative
            // if lambda == 0 , return 0
            // if lambda >0 and <= 10, knuth poisson algorithm
            // if lambda > 10, use transformed rejection method, (Hoermann, 1993)use hoeran's algorithm
            T cur_ele = *(input_tensor + i);
            if (cur_ele < 0) {
                OPENVINO_THROW("RandomPoisson: lambda < 0, rates cannot be negative");
            }

            if (cur_ele == 0) {
                *(output_tensor + i) = 0;
            } else if (cur_ele > 0 && cur_ele < 10) {
                *(output_tensor + i) = knuth_poisson(cur_ele, uniform_sampler);
            } else {
                *(output_tensor + i) = transformed_rejection_method_hoermann(cur_ele, uniform_sampler);
            }
        }
        return generator->get_next_state();
    }

    // TensorFlow: one Philox substream per flat output index (Skip 256 * i from prev_state), one shared U(0,1)
    // converter. A single sequential generator would not match TF RandomPoissonV2 (per-output Skip).
    size_t total_philox_random_calls = 0;
    for (size_t i = 0; i < output_element_count; i++) {
        T cur_ele = *(input_tensor + i);
        if (cur_ele < 0) {
            OPENVINO_THROW("RandomPoisson: lambda < 0, rates cannot be negative");
        }
        if (cur_ele == 0) {
            *(output_tensor + i) = 0;
            continue;
        }
        const auto cell_prev =
            tensorflow_philox_skip(prev_state,
                                   static_cast<uint64_t>(i) * static_cast<uint64_t>(kReserveSamplesPerExecution));
        std::shared_ptr<philox::PhiloxGenerator> cell_generator =
            philox::make_philox_generator(seed, seed2, cell_prev, output_element_count, alignment);
        UniformSampler uniform_sampler(temp_output_buffer_vector.data(),
                                       cell_generator,
                                       converter,
                                       alignment,
                                       &total_philox_random_calls);
        if (cur_ele > 0 && cur_ele < 10) {
            *(output_tensor + i) = knuth_poisson(cur_ele, uniform_sampler);
        } else {
            *(output_tensor + i) = transformed_rejection_method_hoermann(cur_ele, uniform_sampler);
        }
    }

    // UniformSampler uniform_sampler(temp_output_buffer_vector.data(), generator, converter);
    // // For each element in the input tensor, generate a random poisson distributed random number
    // //
    // https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/random-poisson-v2#:~:text=This%20op%20uses,2.%20Addison%20Wesley
    // for (size_t i = 0; i < output_element_count; i++){
    //     // if lambda < 0, throw an exception, rates cannot be negative
    //     // if lambda == 0 , return 0
    //     // if lambda >0 and <= 10, knuth poisson algorithm
    //     // if lambda > 10, use transformed rejection method, (Hoermann, 1993)use hoeran's algorithm
    //     T cur_ele = *(input_tensor + i);
    //     if (cur_ele < 0){
    //         OPENVINO_THROW("RandomPoisson: lambda < 0, rates cannot be negative");
    //     }
    //     if (cur_ele == 0){
    //         *(output_tensor + i) = 0;
    //     }
    //     else if (cur_ele > 0 && cur_ele < 10){
    //         *(output_tensor + i) = knuth_poisson(cur_ele, uniform_sampler);
    //     }
    //     else{
    //         *(output_tensor + i) = transformed_rejection_method_hoermann(cur_ele, uniform_sampler);
    //     }
    // }

    // Return the next state to feed into the generator
    return tensorflow_philox_skip(
        prev_state,
        static_cast<uint64_t>(output_element_count) * static_cast<uint64_t>(kReserveSamplesPerExecution));
}
}  // namespace reference
}  // namespace ov