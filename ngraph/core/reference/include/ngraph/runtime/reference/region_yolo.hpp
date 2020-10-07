//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cmath>

#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            static inline int entry_index(int width,
                                          int height,
                                          int coords,
                                          int classes,
                                          int outputs,
                                          int batch,
                                          int location,
                                          int entry)
            {
                int n = location / (width * height);
                int loc = location % (width * height);
                return batch * outputs + n * width * height * (coords + classes + 1) +
                       entry * width * height + loc;
            }

            template <typename T>
            static inline T sigmoid(float x)
            {
                return static_cast<T>(1.f / (1.f + std::exp(-x)));
            }
            template <typename T>
            static inline void softmax_generic(
                const T* src_data, T* dst_data, int batches, int channels, int height, int width)
            {
                int start = 0;
                for (unsigned int b = 0; b < batches; b++)
                {
                    for (unsigned int i = start; i < height * width; i++)
                    {
                        T max = src_data[b * channels * height * width + i];
                        for (unsigned int c = 0; c < channels; c++)
                        {
                            T val =
                                src_data[b * channels * height * width + c * height * width + i];
                            if (val > max)
                            {
                                max = val;
                            }
                        }

                        T sum = 0;
                        for (unsigned int c = 0; c < channels; c++)
                        {
                            dst_data[b * channels * height * width + c * height * width + i] =
                                std::exp(src_data[b * channels * height * width +
                                                  c * height * width + i] -
                                         max);
                            sum += dst_data[b * channels * height * width + c * height * width + i];
                        }

                        for (unsigned int c = 0; c < channels; c++)
                        {
                            dst_data[b * channels * height * width + c * height * width + i] /= sum;
                        }
                    }
                }
            }

            template <typename T>
            void region_yolo(const T* input,
                             T* output,
                             const Shape& input_shape,
                             const int coords,
                             const int classes,
                             const int regions,
                             const bool do_softmax,
                             const std::vector<int64_t>& mask,
                             const int axis,
                             const int end_axis)
            {
                NGRAPH_CHECK(input_shape.size() == 4);

                const int batches = input_shape[0];
                const int channels = input_shape[1];
                const int height = input_shape[2];
                const int width = input_shape[3];

                const auto mask_size = mask.size();

                std::copy(input, input + shape_size(input_shape), output);

                int num_regions = 0;
                int end_index = 0;

                if (do_softmax)
                {
                    // Region layer (Yolo v2)
                    num_regions = regions;
                    end_index = width * height;
                }
                else
                {
                    // Yolo layer (Yolo v3)
                    num_regions = mask_size;
                    end_index = width * height * (classes + 1);
                }

                const int inputs_size = width * height * num_regions * (classes + coords + 1);

                for (unsigned int b = 0; b < batches; b++)
                {
                    for (unsigned int n = 0; n < num_regions; n++)
                    {
                        int index = entry_index(
                            width, height, coords, classes, inputs_size, b, n * width * height, 0);
                        for (unsigned int i = index; i < index + 2 * width * height; i++)
                        {
                            output[i] = sigmoid<T>(output[i]);
                        }

                        index = entry_index(width,
                                            height,
                                            coords,
                                            classes,
                                            inputs_size,
                                            b,
                                            n * width * height,
                                            coords);
                        for (unsigned int i = index; i < index + end_index; i++)
                        {
                            output[i] = sigmoid<T>(output[i]);
                        }
                    }
                }

                if (do_softmax)
                {
                    int index =
                        entry_index(width, height, coords, classes, inputs_size, 0, 0, coords + 1);
                    int batch_offset = inputs_size / regions;
                    for (unsigned int b = 0; b < batches * regions; b++)
                    {
                        softmax_generic<T>(input + index + b * batch_offset,
                                           output + index + b * batch_offset,
                                           1,
                                           classes,
                                           height,
                                           width);
                    }
                }
            }

        } // namespace reference

    } // namespace runtime

} // namespace ngraph