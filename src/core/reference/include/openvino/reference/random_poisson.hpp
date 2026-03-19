// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/utils/philox_generator.hpp"
#include "openvino/reference/utils/philox_converter.hpp"

#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <cmath>

namespace ov {
namespace reference {

// Knuth's algorithm for generating Poisson distributed random numbers Donald E. Knuth (1969)
template <typename T>
T knuth_poisson(T lambda, 
                  double* temp_output_buffer, 
                  std::shared_ptr<philox::PhiloxGenerator> generator, 
                  std::shared_ptr<philox::PhiloxConverter> converter) {
    auto enlam = std::exp(-lambda);
    T X = 0;
    auto prod = 1.0;
    while (true) {
      const auto& result = generator->random();
      converter->convert(result, 0);
      size_t num_converted = converter->get_converted_elements_count();
      for (size_t i = 0; i < num_converted; i++) {
        auto U_i = temp_output_buffer[i];
        prod *= U_i;
        if (prod > enlam) {
            X += 1;
        } else {
            return static_cast<T>(X);
        }
      }
    }
    return static_cast<T>(X);
}

// Transformed rejection method for generating Poisson distributed random numbers (Hoermann, 1993)
template <typename T>
T transformed_rejection_method_hoermann(T lambda, 
                                          double* temp_output_buffer, 
                                          std::shared_ptr<philox::PhiloxGenerator> generator, 
                                          std::shared_ptr<philox::PhiloxConverter> converter) {
    double slam = std::sqrt(lambda);
    double loglam = std::log(lambda);
    double b = 0.931 + 2.53 * slam;
    double a = -0.059 + 0.02483 * b;
    double invalpha = 1.1239 + 1.1328 / (b - 3.4);
    double vr = 0.9277 - 3.6224 / (b - 2);

    while (true) {
        const auto& result = generator->random();
        converter->convert(result, 0);
        size_t num_converted = converter->get_converted_elements_count();
        for (size_t i = 0; i < num_converted; i+=2) {
            auto U_i = temp_output_buffer[i];
            auto V_i = temp_output_buffer[i+1];
            auto u = U_i - 0.5;
            auto us = 0.5 - std::fabs(u);
            auto k = std::floor((2 * a / us + b) * u + lambda + 0.43);
            if ((us >= 0.07) && (V_i <= vr)) {
                return static_cast<T>(k);
            }
            if ((k < 0) || ((us < 0.013) && (V_i > us))) {
                continue;
            }
            if ((std::log(V_i) + std::log(invalpha) - std::log(a / (us * us) + b)) <= (-lambda + k * loglam - std::lgamma(k + 1))) {
                return static_cast<T>(k);
            }
        }
    }
    return static_cast<T>(0);
}

template <typename T>
std::pair<uint64_t, uint64_t> random_poisson(
    const T* input_tensor, // rates tensor aka lambda values    
    // const T* generator_tensor,
    T* output_tensor,
    const Shape& input_shape,
    // const Shape& generator_shape,
    const Shape& output_shape,
    uint64_t seed,
    uint64_t seed2,
    std::pair<uint64_t, uint64_t> prev_state,
    ov::op::PhiloxAlignment alignment
) {
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

    if (output_element_count != input_element_count){
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
    // Sets up the generator of random numbers and matching converter
    std::shared_ptr<philox::PhiloxGenerator> generator = philox::make_philox_generator(seed, seed2, prev_state, output_element_count, alignment);
    std::shared_ptr<philox::PhiloxConverter> converter = philox::make_philox_converter(temp_output_buffer_ptr, element::f64, philox::ELEMENTS_PER_EXECUTION, min_val_ptr, max_val_ptr, alignment);

    // For each element in the input tensor, generate a random poisson distributed random number
    // https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/random-poisson-v2#:~:text=This%20op%20uses,2.%20Addison%20Wesley
    for (size_t i = 0; i < output_element_count; i++){
        // if lambda < 0, throw an exception, rates cannot be negative
        // if lambda == 0 , return 0
        // if lambda >0 and <= 10, knuth poisson algorithm
        // if lambda > 10, use transformed rejection method, (Hoermann, 1993)use hoeran's algorithm
        T cur_ele = *(input_tensor + i);
        if (cur_ele < 0){
            OPENVINO_THROW("RandomPoisson: lambda < 0, rates cannot be negative");
        }
        if (cur_ele == 0){
            *(output_tensor + i) = 0;
        }
        else if (cur_ele > 0 && cur_ele <= 10){
            *(output_tensor + i) = knuth_poisson(cur_ele, temp_output_buffer_vector.data(), generator, converter);
        }
        else{
            *(output_tensor + i) = transformed_rejection_method_hoermann(cur_ele, temp_output_buffer_vector.data(), generator, converter);
        }
    }

    // Return the next state to feed into the generator
    return generator->get_next_state();
}
}  // namespace reference
}  // namespace ov