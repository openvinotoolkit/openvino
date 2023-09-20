// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <ngraph/shape.hpp>
#include <openvino/reference/broadcast.hpp>
#include <openvino/reference/convert.hpp>
#include <openvino/reference/copy.hpp>
#include <openvino/reference/cum_sum.hpp>
#include <openvino/reference/divide.hpp>
#include <openvino/reference/exp.hpp>
#include <openvino/reference/random_uniform.hpp>
#include <openvino/reference/slice.hpp>

namespace ov {
namespace reference {
namespace multinomial {

template <typename INPUT_T, typename SAMPLES_T, typename OUTPUT_T>
void multinomial(const INPUT_T* input,            // either vector or 2d matrix
                 const Shape& input_shape,        // either (x) or (x,y)
                 const SAMPLES_T* num_samples,    // one-element scalar / vector only with value n
                                                  // with n<=c (c - count of positive values from input 
                                                  // if input is a vector - from whole array
                                                  // otherwise from each channel)
                 const Shape& num_samples_shape,  // (1) 
                 OUTPUT_T* output,                // either vector or 2d matrix
                 const Shape& output_shape,       // (n) if input == vector else (x, n)
                 const bool with_replacement,
                 const bool log_probs,
                 const int64_t global_seed = 0,
                 const int64_t op_seed = 0,
                 const bool validate_args = false  // validate !log_probs => sum per channel == 1,
                                                   // non-zero elements count >= num_samples
) {
    Shape deduced_output_shape;
    if (input_shape.size() > 1) {
        deduced_output_shape.push_back(input_shape[0]);
    }
    if (num_samples_shape.size() > 0) {
        deduced_output_shape.push_back(num_samples[0]);
    }
    size_t total_inputs_elements_count = input_shape.size() > 0;
    size_t total_output_elements_count = deduced_output_shape.size() > 0;

    for (uint64_t i = 0; input_shape.size(); i++) {
        total_inputs_elements_count *= input_shape[i];
    }
    for (uint64_t i = 0; num_samples_shape.size(); i++) {
        total_output_elements_count *= num_samples[i];
    }

    // If probabilities are log probabilities, exponentiate to get normal probabilities
    INPUT_T* input_vals;
    input_vals = (INPUT_T*)malloc(total_inputs_elements_count * sizeof(INPUT_T));
    if (log_probs) {
        exp<INPUT_T>(input, input_vals, total_inputs_elements_count);
    } else {
        copy<INPUT_T>(input, input_vals, total_inputs_elements_count);
    }

    // Create a cdf of probabilties on the last axis, per channel
    void* cdf = malloc(total_inputs_elements_count * sizeof(INPUT_T));
    size_t last_axis = input_shape.size() - 1;
    cumsum<INPUT_T, size_t>(input_vals, &last_axis, static_cast<INPUT_T*>(cdf), input_shape, false, false);

    // Obtain max value from cdf, per channel (from cumsum it is the last element)
    void* max_value_per_channel = malloc(total_inputs_elements_count / input_shape[last_axis] * sizeof(INPUT_T));
    Shape max_value_per_channel_shape(input_shape);
    max_value_per_channel_shape[last_axis] = 1;
    std::vector<int64_t> start{static_cast<int64_t>(input_shape[last_axis] - 1)};
    std::vector<int64_t> step{1};
    std::vector<int64_t> target_axis_vec{static_cast<int64_t>(last_axis)};
    slice(static_cast<char*>(cdf),
          input_shape,
          static_cast<char*>(max_value_per_channel),
          max_value_per_channel_shape,
          sizeof(INPUT_T),
          start,
          step,
          target_axis_vec);

    // Normalize the cdf by dividing all elements by the max value
    void* max_value_elem_divisor = malloc(total_inputs_elements_count * sizeof(INPUT_T));
    ov::AxisSet target_axis_set = ov::AxisSet({last_axis});
    broadcast(static_cast<const char*>(max_value_per_channel),
              static_cast<char*>(max_value_elem_divisor),
              max_value_per_channel_shape,
              input_shape,
              target_axis_set,
              sizeof(INPUT_T));
    divide<INPUT_T>(static_cast<INPUT_T*>(cdf),
                    static_cast<INPUT_T*>(max_value_elem_divisor),
                    static_cast<INPUT_T*>(cdf),
                    total_inputs_elements_count,
                    false);

    // Generate random probability samples
    void* uniform_samples = malloc(sizeof(double) * total_output_elements_count);
    char zero = 0;
    char one = 1;
    std::pair<uint64_t, uint64_t> initial_state(0, 0);
    random_uniform(deduced_output_shape.data(),
                   &zero,
                   &one,
                   static_cast<char*>(uniform_samples),
                   deduced_output_shape,
                   ov::element::f64,
                   global_seed,
                   op_seed,
                   initial_state);

    size_t first_dim_size = input_shape.size() == 2 ? input_shape[0] : 1;
    size_t second_input_dim_size = input_shape.size() == 2 ? input_shape[1] : input_shape[0];
    size_t second_output_dim_size = input_shape.size() == 2 ? num_samples[0] : input_shape[0];

    // Iterate over each channel in uniform samples
    void* output_samples = malloc(sizeof(SAMPLES_T) * total_output_elements_count);
    for (size_t i = 0; i < first_dim_size; ++i) {
        for (size_t j = 0; j < second_output_dim_size; ++j) {
            // Iterate over cdf to find the index for a given sample
            double sample_value = ((double*)uniform_samples)[i * first_dim_size + j];
            size_t selected_class_idx;
            for (size_t l = 0; l < second_input_dim_size; ++l) {
                if (sample_value <= ((INPUT_T*)cdf)[i * first_dim_size + l]) {
                    ((SAMPLES_T*)output_samples)[i * second_output_dim_size + j] = l;
                    selected_class_idx = l;
                    break;
                }
            }
            // Additional step - change probability of a given class to 0, and update the cdf
            if (with_replacement) {
                INPUT_T class_probability = input_vals[i * first_dim_size + selected_class_idx];
                INPUT_T divisor = ((INPUT_T*)cdf)[i * first_dim_size + second_input_dim_size - 1] - class_probability;
                for (size_t k = 0; k < second_input_dim_size - 1; ++k) {
                    if (k >= selected_class_idx) {
                        ((INPUT_T*)cdf)[i * first_dim_size + k] -= class_probability;
                    }
                    ((INPUT_T*)cdf)[i * first_dim_size + k] /= divisor;
                }
                // This should always be true - verify if line needed
                // cdf[i * first_dim_size + second_input_dim_size - 1] = 1;
            }
        }
    }
    // Finally convert the samples to the requested data type
    convert<SAMPLES_T, OUTPUT_T>(static_cast<SAMPLES_T*>(output_samples), output, total_output_elements_count);

    free(output_samples);
    free(uniform_samples);
    free(max_value_elem_divisor);
    free(max_value_per_channel);
    free(cdf);
    free(input_vals);
}
}  // namespace multinomial
}  // namespace reference
}  // namespace ov
