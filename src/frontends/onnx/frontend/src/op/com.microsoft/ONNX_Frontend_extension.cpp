#include "ONNX_Frontend_extension.hpp"
#include <iostream>

namespace onnx_opset {
    namespace op {
        void Pad(const float* input_data, const int64_t* pads_data, float* output_data, const std::vector<int64_t>& input_shape, const std::vector<int64_t>& output_shape, const std::string& mode, float constant_value) {
            int64_t input_rank = input_shape.size();
            int64_t output_rank = output_shape.size();
            int64_t num_axes = input_rank;  // Number of axes to pad
            int64_t channels = input_shape[1];  // Number of channels

            // Iterate over each element in the output tensor
            for (int64_t output_index = 0; output_index < output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]; ++output_index) {
                // Convert output index to corresponding input index
                std::vector<int64_t> input_index(input_rank);
                int64_t remainder = output_index;
                for (int64_t i = output_rank - 1; i >= 0; --i) {
                    input_index[i] = remainder % input_shape[i];
                    remainder /= input_shape[i];
                }

                // Check if the current element is within the original input shape
                bool within_input = true;
                for (int64_t i = 0; i < num_axes; ++i) {
                    if (input_index[i] < 0 || input_index[i] >= input_shape[i]) {
                        within_input = false;
                        break;
                    }
                }

                // Perform padding based on the selected mode
                if (within_input) {
                    // Calculate input index based on padding values
                    std::vector<int64_t> padded_input_index(input_rank);
                    for (int64_t i = 0; i < num_axes; ++i) {
                        padded_input_index[i] = input_index[i] - pads_data[i];
                    }

                    // Convert padded input index to linear index
                    int64_t input_index_linear = 0;
                    int64_t multiplier = 1;
                    for (int64_t i = input_rank - 1; i >= 0; --i) {
                        input_index_linear += padded_input_index[i] * multiplier;
                        multiplier *= input_shape[i];
                    }

                    // Copy value from the original input to the output tensor
                    output_data[output_index] = input_data[input_index_linear];
                } else {
                    // Perform padding based on the selected mode
                    if (mode == "constant") {
                        output_data[output_index] = constant_value;
                    } else if (mode == "reflect") {
                        // Calculate reflected index
                        std::vector<int64_t> reflected_index(input_rank);
                        for (int64_t i = 0; i < num_axes; ++i) {
                            int64_t pad = pads_data[i];
                            int64_t input_dim = input_shape[i];
                            int64_t input_index_i = input_index[i] - pads_data[i];
                            if (input_index_i < 0) {
                                input_index_i = -input_index_i;
                            }
                            if (input_index_i >= input_dim) {
                                input_index_i = 2 * input_dim - input_index_i - 2;
                            }
                            reflected_index[i] = input_index_i;
                        }

                        // Convert reflected index to linear index
                        int64_t input_index_linear = 0;
                        int64_t multiplier = 1;
                        for (int64_t i = input_rank - 1; i >= 0; --i) {
                            input_index_linear += reflected_index[i] * multiplier;
                            multiplier *= input_shape[i];
                        }

                        // Copy value from the original input to the output tensor
                        output_data[output_index] = input_data[input_index_linear];
                    } else if (mode == "edge") {
                        // Calculate clamped index
                        std::vector<int64_t> clamped_index(input_rank);
                        for (int64_t i = 0; i < num_axes; ++i) {
                            int64_t pad = pads_data[i];
                            int64_t input_dim = input_shape[i];
                            int64_t input_index_i = input_index[i] - pads_data[i];
                            if (input_index_i < 0) {
                                input_index_i = 0;
                            }
                            if (input_index_i >= input_dim) {
                                input_index_i = input_dim - 1;
                            }
                            clamped_index[i] = input_index_i;
                        }

                        // Convert clamped index to linear index
                        int64_t input_index_linear = 0;
                        int64_t multiplier = 1;
                        for (int64_t i = input_rank - 1; i >= 0; --i) {
                            input_index_linear += clamped_index[i] * multiplier;
                            multiplier *= input_shape[i];
                        }

                        // Copy value from the original input to the output tensor
                        output_data[output_index] = input_data[input_index_linear];
                    }
                }
            }
        }
    }
}
