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

#include <cstring>
#include <vector>

#include "ngraph/check.hpp"
#include "ngraph/runtime/reference/slice.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            class Slice
            {
            public:
                Slice(const char* in_data,
                      char* out_data,
                      const Shape& source_shape,
                      const Coordinate& lower_bounds,
                      const Coordinate& upper_bounds,
                      const Strides& strides,
                      size_t elem_size)
                    : m_in_data{in_data}
                    , m_out_data{out_data}
                    , m_source_shape{source_shape}
                    , m_lower_bounds{lower_bounds}
                    , m_upper_bounds{upper_bounds}
                    , m_strides{strides}
                    , m_elem_size{elem_size}
                    , m_memory_strides{memory_stirdes(source_shape)}
                    , m_memory_index{index(lower_bounds, source_shape)}
                {
                }

                void execute() { loop(0); }
            private:
                struct GoDeeper
                {
                    static void run(Slice& r, size_t dim) { r.loop(dim); }
                };

                struct DoMemCpy
                {
                    static void run(Slice& r, size_t)
                    {
                        std::memcpy(r.m_out_data,
                                    r.m_in_data + r.m_memory_index * r.m_elem_size,
                                    r.m_elem_size);
                        r.m_out_data += r.m_elem_size;
                    }
                };

                static std::vector<size_t> memory_stirdes(const Shape& s)
                {
                    std::vector<size_t> mem_strides(s.size(), 1);
                    for (size_t i = mem_strides.size() - 1; i-- > 0;)
                    {
                        mem_strides[i] = mem_strides[i + 1] * s[i + 1];
                    }
                    return mem_strides;
                }

                static size_t index(const Coordinate& c, const Shape& s) noexcept
                {
                    size_t index = 0;
                    size_t stride = 1;
                    size_t const padding = c.size() - s.size();

                    for (size_t axis = s.size(); axis-- > 0;)
                    {
                        if (s[axis] > 1)
                        {
                            index += c[axis + padding] * stride;
                            stride *= s[axis];
                        }
                    }

                    return index;
                }

                template <typename Algo = GoDeeper>
                void loop(size_t dim)
                {
                    if (std::is_same<Algo, GoDeeper>::value && dim == m_source_shape.size() - 1)
                    {
                        loop<DoMemCpy>(dim);
                        return;
                    }
                    const size_t dim_size = upper_bounds(dim) - lower_bounds(dim);
                    const auto start_mem_index = m_memory_index;
                    for (size_t d = 0; d < dim_size; d += m_strides[dim])
                    {
                        Algo::run(*this, dim + 1);
                        m_memory_index += m_memory_strides[dim] * m_strides[dim];
                    }
                    m_memory_index = start_mem_index;
                }

                size_t upper_bounds(size_t index) const
                {
                    return m_upper_bounds[index] < 0 ? m_source_shape[index] + m_upper_bounds[index]
                                                     : m_upper_bounds[index];
                }
                size_t lower_bounds(size_t index) const
                {
                    return m_lower_bounds[index] < 0 ? m_source_shape[index] + m_lower_bounds[index]
                                                     : m_lower_bounds[index];
                }

                const char* const m_in_data;
                char* m_out_data;

                const Shape& m_source_shape;
                const Coordinate& m_lower_bounds;
                const Coordinate& m_upper_bounds;
                const Strides& m_strides;
                const size_t m_elem_size{0};

                const std::vector<size_t> m_memory_strides;
                size_t m_memory_index{0};
            };

            void slice(const char* arg,
                       char* out,
                       const Shape& arg_shape,
                       const Coordinate& lower_bounds,
                       const Coordinate& upper_bounds,
                       const Strides& strides,
                       const Shape& out_shape,
                       size_t elem_size)
            {
                const CoordinateTransform input_transform(
                    arg_shape, lower_bounds, upper_bounds, strides);

                const CoordinateTransform output_transform(out_shape);

                NGRAPH_CHECK(shape_size(input_transform.get_target_shape()) ==
                             shape_size(output_transform.get_target_shape()));

                Slice s{arg, out, arg_shape, lower_bounds, upper_bounds, strides, elem_size};
                s.execute();
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
