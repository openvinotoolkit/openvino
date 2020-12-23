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

#include "ngraph/coordinate_range.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/gather_nd.hpp"
#include "utils/span.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace
            {
                template <typename Container>
                Shape to_shape(const Container& c)
                {
                    return Shape(begin(c), end(c));
                }

                template <typename Container>
                std::vector<size_t>
                    join(const Container& c1, const Container& c2, const Container& c3)
                {
                    using container_value_type =
                        typename std::remove_cv<typename Container::value_type>::type;
                    static_assert(std::is_same<container_value_type, size_t>::value,
                                  "Expect same type in container");
                    std::vector<size_t> ret;
                    ret.reserve(c1.size() + c2.size() + c3.size());
                    std::copy(begin(c1), end(c1), std::back_inserter(ret));
                    std::copy(begin(c2), end(c2), std::back_inserter(ret));
                    std::copy(begin(c3), end(c3), std::back_inserter(ret));
                    return ret;
                }

                const auto only_one = [] { return coordinates::index(Shape{1}); };
            } // namespace
            template <typename T, typename U>
            void gather(const T* const params,
                        const U* const indices,
                        T* const out,
                        const Shape& params_shape,
                        const Shape& indices_shape,
                        const Shape& out_shape,
                        size_t axis)
            {
                using std::next;
                assert(std::memset(out, 0, shape_size(out_shape) * sizeof(T)));

                const auto params_axes_part = span(params_shape).subspan(0, axis);

                NGRAPH_CHECK(params_shape.size() >= axis, "Not enough axes in param_shape.");

                const auto remainder_part_shape = span(params_shape).subspan(axis + 1);

                const auto found_out_shape =
                    join(params_axes_part, span(indices_shape), remainder_part_shape);

                NGRAPH_CHECK(found_out_shape == out_shape,
                             "Output shape mismatch with calculations");

                const auto batch_shape = span(params_shape).subspan(axis);

                const auto batch_size = shape_size(batch_shape);

                const auto copy_size = shape_size(remainder_part_shape);

                const size_t copy_round_in_batch =
                    indices_shape.size() > 1
                        ? shape_size(span(indices_shape.data(), indices_shape.size() - 1))
                        : 1;
                const size_t round_batch_offset = indices_shape.empty() ? 1 : indices_shape.back();

                auto dst = out;

                auto gather_range = params_axes_part.empty()
                                        ? only_one()
                                        : coordinates::index(to_shape(params_axes_part));
                for (auto i : gather_range)
                {
                    auto batch_index = i.begin_index;
                    for (size_t batch = 0; batch != i.element_number;
                         batch_index += i.step, ++batch)
                    {
                        const auto batch_offset = batch_index * batch_size;
                        assert(batch_offset < shape_size(params_shape));
                        for (size_t round = 0; round != copy_round_in_batch; ++round)
                        {
                            const U* input_indices = indices + round * round_batch_offset;
                            const auto indices_no =
                                indices_shape.empty() ? 1 : indices_shape.back();

                            assert(!batch_shape.empty());
                            for (size_t ii = 0; ii != indices_no; ++ii)
                            {
                                const auto positive_input_index =
                                    input_indices[ii] < 0 ? batch_shape.front() + input_indices[ii]
                                                          : input_indices[ii];

                                const auto src_offset =
                                    batch_offset + copy_size * positive_input_index;

                                const auto src_begin = next(params, src_offset);
                                const auto src_end = next(src_begin, copy_size);

                                std::copy(src_begin, src_end, dst);
                                dst += copy_size;
                            }
                        }
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
