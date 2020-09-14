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
                auto max_seq_len = data_shape[0];
                auto batch_size = data_shape[1];
                auto class_count = data_shape[2];

                CoordinateTransform out_transform = CoordinateTransform(out_shape);
                CoordinateTransform data_transform = CoordinateTransform(data_shape);
                CoordinateTransform seq_masks_transform = CoordinateTransform(sequence_masks_shape);

                // final sequences don't have to fill the whole output, elements that don't store
                // information are set to -1
                std::fill(out, out + out_shape.size(), static_cast<T>(-1.0));

                for (unsigned int seq_ind = 0; seq_ind < max_seq_len; seq_ind++)
                {
                    for (unsigned int batch_ind = 0; batch_ind < batch_size; batch_ind++)
                    {
                        auto data_index = data_transform.index({batch_ind, seq_ind, 0});
                        auto mask_index = seq_masks_transform.index({batch_ind, seq_ind});
                        // first 0 marks the end of a sequence
                        if (sequence_masks[mask_index] != static_cast<T>(1))
                        {
                            continue;
                        }
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph