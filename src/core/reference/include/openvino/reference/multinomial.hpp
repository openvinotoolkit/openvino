// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/reference/broadcast.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/reference/copy.hpp"
#include "openvino/reference/cum_sum.hpp"
#include "openvino/reference/divide.hpp"
#include "openvino/reference/exp.hpp"
#include "openvino/reference/random_uniform.hpp"
#include "openvino/reference/slice.hpp"

namespace ov {
namespace reference {
namespace multinomial {
/**
 * @brief Multinomial operation creates a sequence of indices of classes sampled from the multinomial distribution.
 *
 * @tparam T Data type of the probs' values.
 * @tparam U Data type of num_samples' values.
 * @tparam V Data type of output's values.
 * @param probs Input tensor containing at each index poisition probability/log probability of sampling a given class.
 * @param probs_shape Shape of the 'probs' tensor.
 * @param num_samples Scalar or 1D tensor with a single value that determines the number of samples to generate per
 * batch.
 * @param num_samples_shape Shape of the 'num_samples' tensor.
 * @param output Output tensor for the generated class indices.
 * @param output_shape Shape of the 'output' tensor.
 * @param with_replacement Boolean that determines whether a sampled class can appear more than once in the output.
 * @param log_probs Boolean that determines whether to treat input probabilities as log probabilities.
 * @param global_seed First seed value (key) of Philox random number generation algorithm. (See RandomUniform for
 * details)
 * @param op_seed Second seed value (counter) of Philox random number generation algorithm. (See RandomUniform for
 * details)
 */
template <typename T, typename U, typename V>
void multinomial(const T* probs,
                 const Shape& probs_shape,
                 const U* num_samples,
                 const Shape& num_samples_shape,
                 V* output,
                 const Shape& output_shape,
                 const bool with_replacement,
                 const bool log_probs,
                 const uint64_t global_seed,
                 const uint64_t op_seed) {
    const auto total_inputs_elements_count = shape_size<Shape>(probs_shape);
    const auto total_output_elements_count = shape_size<Shape>(output_shape);

    // If probabilities are log probabilities, exponentiate to get normal probabilities
    std::vector<T> input_vals(total_inputs_elements_count);
    if (log_probs) {
        exp(probs, input_vals.data(), total_inputs_elements_count);
    } else {
        copy(probs, input_vals.data(), total_inputs_elements_count);
    }

    // Create a cdf of probabilties on the last axis, per batch. Note cumsum exclusive  == false
    std::vector<T> cdf(total_inputs_elements_count);
    const auto last_axis = probs_shape.size() - 1;
    cumsum(input_vals.data(), last_axis, cdf.data(), probs_shape, false, false);

    // Obtain max value from cdf, per batch (from cumsum it is the last element)
    std::vector<T> max_value_per_batch(total_inputs_elements_count / probs_shape[last_axis]);
    Shape max_value_per_batch_shape(probs_shape);
    max_value_per_batch_shape[last_axis] = 1;
    const std::vector<int64_t> start{static_cast<int64_t>(probs_shape[last_axis] - 1)};
    const std::vector<int64_t> step{1};
    const std::vector<int64_t> target_axis_vec{static_cast<int64_t>(last_axis)};
    slice(reinterpret_cast<const char*>(cdf.data()),
          probs_shape,  // == cdf shape
          reinterpret_cast<char*>(max_value_per_batch.data()),
          max_value_per_batch_shape,
          sizeof(T),
          start,
          step,
          target_axis_vec);

    // Normalize the cdf by dividing all elements by the max value in each batch
    std::vector<T> max_value_per_batch_divisor(total_inputs_elements_count);
    ov::AxisSet target_axis_set = ov::AxisSet({last_axis});
    broadcast(reinterpret_cast<const char*>(max_value_per_batch.data()),
              reinterpret_cast<char*>(max_value_per_batch_divisor.data()),
              max_value_per_batch_shape,
              probs_shape,  // expand to original shape (expands last dim)
              target_axis_set,
              sizeof(T));
    divide(cdf.data(), max_value_per_batch_divisor.data(), cdf.data(), total_inputs_elements_count, false);

    // Generate random probability samples
    std::vector<T> uniform_samples(total_output_elements_count);
    const T zero = 0;
    const T one = 1;
    const ov::Shape output_shape_shape{output_shape.size()};
    const std::vector<uint64_t> output_shape_u64(output_shape.begin(), output_shape.end());
    const std::pair<uint64_t, uint64_t> initial_state(0, 0);
    random_uniform(output_shape_u64.data(),
                   reinterpret_cast<const char*>(&zero),
                   reinterpret_cast<const char*>(&one),
                   reinterpret_cast<char*>(uniform_samples.data()),
                   output_shape_shape,
                   ov::element::from<T>(),
                   global_seed,
                   op_seed,
                   initial_state);

    auto batch_size = probs_shape.size() == 2 ? static_cast<size_t>(probs_shape[0]) : static_cast<size_t>(1);
    auto class_size =
        probs_shape.size() == 2 ? static_cast<size_t>(probs_shape[1]) : static_cast<size_t>(probs_shape[0]);
    auto samples_size = static_cast<size_t>(num_samples[0]);

    // Iterate over each channel in uniform samples
    std::vector<U> output_samples(total_output_elements_count);
    for (size_t i = 0; i < batch_size * samples_size; i += samples_size) {
        for (size_t j = 0; j < samples_size; ++j) {
            // Iterate over cdf to find the index for a given sample
            // If no class found (all have 0 probability), selects last - undefined behavior
            auto i_translated = i / samples_size * class_size;
            auto selected_class_idx = class_size;
            auto sample_value = uniform_samples[i + j];
            for (size_t k = 0; k < class_size; ++k) {
                if (sample_value <= cdf[i_translated + k]) {
                    output_samples[i + j] = static_cast<U>(k);
                    selected_class_idx = k;
                    break;
                }
            }
            // Additional step without replacement - change probability of a given class to 0, and update the cdf
            if (!with_replacement) {
                T class_probability = selected_class_idx ? cdf[i_translated + selected_class_idx] -
                                                               cdf[i_translated + selected_class_idx - 1]
                                                         : cdf[i_translated];
                T divisor = 1 - class_probability;
                for (size_t k = 0; k < class_size; ++k) {
                    if (k >= selected_class_idx) {
                        cdf[i_translated + k] -= class_probability;
                    }
                    cdf[i_translated + k] /= divisor;
                }
            }
        }
    }
    // Finally convert the samples to the requested data type
    convert<U, V>(output_samples.data(), output, total_output_elements_count);
}
}  // namespace multinomial
}  // namespace reference

namespace op {
namespace multinomial {
namespace validate {
void input_types(const Node* op);
}  // namespace validate
}  // namespace multinomial
}  // namespace op
}  // namespace ov
