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

#include <cmath>
#include <numeric>
#include <stdio.h>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/gather_tree.hpp"

using namespace ngraph;

static size_t _asIndex(const char* source, const element::Type& element_type)
{
    // According to the GatherTree op specification only I32 and FP32 precisions are supported.
    switch (element_type)
    {
    case element::Type_t::f32:
    {
        float tmpBuff = 0.f;
        memcpy(&tmpBuff, source, sizeof(float));
        return tmpBuff;
    }
    case element::Type_t::i32:
    {
        int32_t tmpBuff = 0;
        memcpy(&tmpBuff, source, sizeof(int32_t));
        return tmpBuff;
    }
    default:
    {
        throw ngraph_error(std::string("Unsupported input data type: ") +
                           element_type.get_type_name());
    }
    }
}

// This is an implementation of the algorithm from the tensorflow 1.5 sources.
void runtime::reference::gather_tree(const char* step_ids,
                                     const char* parent_ids,
                                     const char* max_seq_len,
                                     const char* end_token,
                                     char* out,
                                     const Shape& step_ids_shape,
                                     const Shape& parent_ids_shape,
                                     const Shape& max_seq_len_shape,
                                     const Shape& end_token_shape,
                                     const element::Type& element_type)
{
    if (step_ids_shape != parent_ids_shape)
    {
        throw ngraph_error("step_ids shape and parent_ids shape must be the same");
    }
    if (step_ids_shape.size() != 3)
    {
        throw ngraph_error("step_ids must be a 3-tensor");
    }
    if (!is_vector(max_seq_len_shape))
    {
        throw ngraph_error("max_seq_len must be a vector");
    }
    if (!is_scalar(end_token_shape))
    {
        throw ngraph_error("end_token must be a scalar");
    }

    const size_t max_time = step_ids_shape.at(0);
    const size_t batch_size = step_ids_shape.at(1);
    const size_t beam_width = step_ids_shape.at(2);

    const size_t elem_size = element_type.size();

    if (max_seq_len_shape.front() != batch_size)
    {
        throw ngraph_error("max_seq_len must have size of BATCH_SIZE");
    }

    ngraph::CoordinateTransform cordinate_transform(step_ids_shape);

    for (const auto& coord : cordinate_transform)
    {
        memcpy(out + cordinate_transform.index(coord) * elem_size, end_token, elem_size);
    }

    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        for (size_t beam = 0; beam < beam_width; ++beam)
        {
            const size_t max_seq_in_beam =
                std::min(max_time, _asIndex(max_seq_len + batch * elem_size, element_type));

            if (max_seq_in_beam == 0)
            {
                continue;
            }

            auto offset = cordinate_transform.index({max_seq_in_beam - 1, batch, beam}) * elem_size;

            memcpy(out + offset, step_ids + offset, elem_size);

            size_t parent = _asIndex(parent_ids + offset, element_type);

            for (size_t level = max_seq_in_beam - 1; level-- > 0;)
            {
                memcpy(out + cordinate_transform.index({level, batch, beam}) * elem_size,
                       step_ids + cordinate_transform.index({level, batch, parent}) * elem_size,
                       elem_size);

                parent = _asIndex(parent_ids +
                                      cordinate_transform.index({level, batch, parent}) * elem_size,
                                  element_type);
            }

            bool finished = false;
            for (size_t time = 0; time < max_seq_in_beam; ++time)
            {
                if (finished)
                {
                    memcpy(out + cordinate_transform.index({time, batch, beam}) * elem_size,
                           end_token,
                           elem_size);
                }
                else if (_asIndex(out + cordinate_transform.index({time, batch, beam}) * elem_size,
                                  element_type) == _asIndex(end_token, element_type))
                {
                    finished = true;
                }
            }
        }
    }
}