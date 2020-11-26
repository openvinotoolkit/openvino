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
#include "ngraph/runtime/reference/reverse.hpp"

using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            class Reverse
            {
            public:
                Reverse(const char* in_data,
                        char* out_data,
                        const Shape& source_shape,
                        const AxisSet& reversed_axes,
                        size_t elem_size)
                    : m_in_data{in_data}
                    , m_out_data{out_data}
                    , m_source_shape{source_shape}
                    , m_reversed_axes{axes_order(source_shape, reversed_axes)}
                    , m_elem_size{elem_size}
                    , m_memory_strides{memory_stirdes(source_shape)}
                    , m_memory_index{0}
                {
                }

                void execute() { loop(0); }
            private:
                struct GoDeeper
                {
                    static void run(Reverse& r, size_t dim) { r.loop(dim); }
                };

                struct DoMemCpy
                {
                    static void run(Reverse& r, size_t)
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

                static std::vector<int> axes_order(const Shape& s, const AxisSet& reverse_axes)
                {
                    std::vector<int> order(s.size(), 1);
                    for (auto i : reverse_axes)
                    {
                        order[i] = -1;
                    }
                    return order;
                }

                template <typename Algo = GoDeeper>
                void loop(size_t dim)
                {
                    if (std::is_same<Algo, GoDeeper>::value && dim == m_source_shape.size() - 1)
                    {
                        loop<DoMemCpy>(dim);
                        return;
                    }
                    const size_t dim_size = m_source_shape[dim];
                    const auto start_mem_index = m_memory_index;
                    if (m_reversed_axes[dim] < 0)
                    {
                        m_memory_index += (dim_size - 1) * m_memory_strides[dim];
                    }
                    for (size_t d = 0; d < dim_size; ++d)
                    {
                        Algo::run(*this, dim + 1);
                        m_memory_index += m_memory_strides[dim] * m_reversed_axes[dim];
                    }
                    m_memory_index = start_mem_index;
                }

                const char* const m_in_data;
                char* m_out_data;

                const Shape& m_source_shape;
                const std::vector<int> m_reversed_axes;
                const size_t m_elem_size{0};

                const std::vector<size_t> m_memory_strides;
                size_t m_memory_index{0};
            };

            void reverse(const char* arg,
                         char* out,
                         const Shape& arg_shape,
                         const Shape& out_shape,
                         const AxisSet& reversed_axes,
                         size_t elem_size)
            {
                NGRAPH_CHECK(shape_size(arg_shape) == shape_size(out_shape));

                const bool nothing_to_revers = reversed_axes.empty();
                if (nothing_to_revers)
                {
                    std::memcpy(out, arg, shape_size(arg_shape) * elem_size);
                    return;
                }

                Reverse r{arg, out, arg_shape, reversed_axes, elem_size};
                r.execute();
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
