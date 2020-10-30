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

#include <algorithm>
#include <limits>
#include <vector>
#include "ngraph/coordinate_transform.hpp"
namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void ctc_greedy_decoder(const T* data,
                                    const T* sequence_masks,
                                    T* out,
                                    const Shape& data_shape,
                                    const Shape& sequence_masks_shape,
                                    const Shape& out_shape,
                                    const bool ctc_merge_repeated)
            {
                const auto max_seq_len = data_shape[0];
                const auto batch_size = data_shape[1];
                const auto class_count = data_shape[2];
                const auto blank_index = class_count - 1;

                CoordinateTransform out_transform = CoordinateTransform(out_shape);
                CoordinateTransform data_transform = CoordinateTransform(data_shape);
                CoordinateTransform seq_masks_transform = CoordinateTransform(sequence_masks_shape);

                // final sequences don't have to fill the whole output, elements that don't store
                // information are set to -1

                std::vector<T> tmp_out(shape_size(out_shape));
                std::fill(tmp_out.begin(), tmp_out.end(), static_cast<T>(-1.0));

                for (unsigned int batch_ind = 0; batch_ind < batch_size; batch_ind++)
                {
                    T previous_class_index = static_cast<T>(-1);
                    auto out_index = out_transform.index({batch_ind, 0, 0, 0});
                    for (unsigned int seq_ind = 0; seq_ind < max_seq_len; seq_ind++)
                    {
                        auto data_index = data_transform.index({seq_ind, batch_ind, 0});
                        auto mask_index = seq_masks_transform.index({seq_ind, batch_ind});

                        // first 0 marks the end of a sequence
                        if (seq_ind && sequence_masks[mask_index] == T{0})
                        {
                            break;
                        }

                        auto class_index = data + data_index;
                        auto class_max_element =
                            std::max_element(class_index, class_index + class_count);
                        unsigned int max_class_ind = std::distance(class_index, class_max_element);
                        if (!(previous_class_index == max_class_ind && ctc_merge_repeated) &&
                            max_class_ind < blank_index)
                        {
                            tmp_out[out_index++] = max_class_ind;
                        }
                        previous_class_index = max_class_ind;
                    }
                }
                std::copy(tmp_out.begin(), tmp_out.end(), out);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
