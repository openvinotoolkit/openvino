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

#include <numeric>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            //This is a implementation of the algorithm from the tensorflow 1.5 sources.
            template<typename T>
            void gather_tree(const T* step_ids,
                             const T* parent_ids,
                             const T* max_seq_len,
                             const T* end_token,
                             T* out,
                             const Shape& step_ids_shape,
                             const Shape& parent_ids_shape,
                             const Shape& max_seq_len_shape,
                             const Shape& end_token_shape)
            {
                if(step_ids_shape != parent_ids_shape) {
                    throw ngraph_error("step_ids shape and parent_ids shape must be the same");
                }
                if(step_ids_shape.size() != 3) {
                    throw ngraph_error("step_ids must be a 3-tensor");
                }
                if(!is_vector(max_seq_len_shape)) {
                    throw ngraph_error("max_seq_len must be a vector");
                }
                if(!is_scalar(end_token_shape)) {
                    throw ngraph_error("end_token must be a scalar");
                }

                const size_t max_time = step_ids_shape.at(0);
                const size_t batch_size = step_ids_shape.at(1);
                const size_t beam_width = step_ids_shape.at(2);

                if(max_seq_len_shape.front() != batch_size) {
                    throw ngraph_error("max_seq_len must have size of BATCH_SIZE");
                }

                ngraph::CoordinateTransform cordinate_transform(step_ids_shape);

                for(const auto& coord : cordinate_transform) {
                    out[cordinate_transform.index(coord)] = *end_token;
                }

                for(size_t batch = 0; batch <  batch_size; ++batch) {

                    for (size_t beam = 0; beam < beam_width; ++beam) {
                        const size_t max_seq_in_beam = std::min(max_time, size_t(max_seq_len[batch]));

                        if (max_seq_in_beam <= 0) {
                            continue;
                        }

                        auto indx = cordinate_transform.index({max_seq_in_beam - 1, batch, beam});

                        out[indx] = step_ids[indx];

                        size_t parent = parent_ids[indx];

                        for (size_t level = max_seq_in_beam - 1; level-- > 0;) {
                            out[cordinate_transform.index({level, batch, beam})] = step_ids[cordinate_transform.index({level, batch, parent})];

                            parent = parent_ids[cordinate_transform.index({level, batch, parent})];
                        }

                        bool finished = false;
                        for (size_t time = 0; time < max_seq_in_beam; ++time) {
                            if (finished) {
                                out[cordinate_transform.index({time, batch, beam})] = *end_token;
                            } else if (out[cordinate_transform.index({time, batch, beam})] == *end_token) {
                                finished = true;
                            }
                        }
                    }
                }
            }
        }
    }
}