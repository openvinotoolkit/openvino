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
#include <cstddef>

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

            static inline float logistic_activate(float x) { return 1.f / (1.f + std::exp(-x)); }
            template <typename T>
            static inline void
                softmax_generic(const T* src_data, T* dst_data, int B, int C, int H, int W)
            {
                int start = 0;
                for (unsigned int b = 0; b < B; b++)
                {
                    for (unsigned int i = start; i < H * W; i++)
                    {
                        float max = src_data[b * C * H * W + i];
                        for (unsigned int c = 0; c < C; c++)
                        {
                            float val = src_data[b * C * H * W + c * H * W + i];
                            if (val > max)
                            {
                                max = val;
                            }
                        }

                        float sum = 0;
                        for (unsigned int c = 0; c < C; c++)
                        {
                            dst_data[b * C * H * W + c * H * W + i] =
                                std::exp(src_data[b * C * H * W + c * H * W + i] - max);
                            sum += dst_data[b * C * H * W + c * H * W + i];
                        }

                        for (unsigned int c = 0; c < C; c++)
                        {
                            dst_data[b * C * H * W + c * H * W + i] /= sum;
                        }
                    }
                }
            }

            template <typename T>
            void region_yolo(const T* input,
                             const Shape& input_shape,
                             const int coords,
                             const int classes,
                             const int regions,
                             const bool do_softmax,
                             const std::vector<int64_t>& mask,
                             const int axis,
                             const int end_axis,
                             const std::vector<float>& anchors,
                             T* output,
                             const Shape& output_shape)
            {
                NGRAPH_CHECK(input_shape.size() == 4);

                const int B = input_shape[0];
                const int C = input_shape[1];
                const int H = input_shape[2];
                const int W = input_shape[3];

                const auto mask_size = mask.size();
                const int tensor_size = B * C * H * W;
                for (unsigned int i = 0; i < tensor_size; i++)
                {
                    output[i] = input[i];
                }

                int num_regions = 0;
                int end_index = 0;

                if (do_softmax)
                {
                    // Region layer (Yolo v2)
                    num_regions = regions;
                    end_index = W * H;
                }
                else
                {
                    // Yolo layer (Yolo v3)
                    num_regions = mask_size;
                    end_index = W * H * (classes + 1);
                }

                const int inputs_size = W * H * num_regions * (classes + coords + 1);

                for (unsigned int b = 0; b < B; b++)
                {
                    for (unsigned int n = 0; n < num_regions; n++)
                    {
                        int index =
                            entry_index(W, H, coords, classes, inputs_size, b, n * W * H, 0);
                        for (unsigned int i = index; i < index + 2 * W * H; i++)
                        {
                            output[i] = logistic_activate(output[i]);
                        }

                        index =
                            entry_index(W, H, coords, classes, inputs_size, b, n * W * H, coords);
                        for (unsigned int i = index; i < index + end_index; i++)
                        {
                            output[i] = logistic_activate(output[i]);
                        }
                    }
                }

                if (do_softmax)
                {
                    int index = entry_index(W, H, coords, classes, inputs_size, 0, 0, coords + 1);
                    int batch_offset = inputs_size / regions;
                    for (unsigned int b = 0; b < B * regions; b++)
                    {
                        softmax_generic<T>(input + index + b * batch_offset,
                                           output + index + b * batch_offset,
                                           1,
                                           classes,
                                           H,
                                           W);
                    }
                }
            }

        } // namespace reference

    } // namespace runtime

} // namespace ngraph