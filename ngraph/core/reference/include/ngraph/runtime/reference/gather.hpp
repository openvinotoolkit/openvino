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

#include <chrono>
#include <iostream>
#include <numeric>

#include "ngraph/coordinate_range.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/gather_nd.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace v0
            {
                // Implement gather by calling gather_nd on sub-problems
                // # prepare constant shapes for tensors used for sub problems
                // indices'.shape  = indices.shape[-1] + [1]
                // params'.shape = params.shape[axis:]
                // out'.shape = params'.shape
                // out'.shape[0] = indices.shape[-1]
                // # call sub-problems
                // foreach (params_index, out_index) in outer "axis" dimensions
                //     # params_prime is shared by inner loop
                //     params' = param[params_index] # rank(params') == rank(params) - axis
                //     foreach indices_index in outer N-1 dimensions
                //         indices' = indices[indices_index] # rank(indices') == 2
                //         out_index = out_index + indices_index
                //         out' = out[out_index] # rank(out') == rank(params')
                //         gather_nd(params', indices'', out')
                template <typename T, typename U>
                void gather(const T* params,
                            const U* indices,
                            T* out,
                            const Shape& params_shape,
                            const Shape& indices_shape,
                            const Shape& out_shape,
                            size_t axis)
                {
                    using namespace std;
                    // prepare shape of params_prime (remove first "axis" dimensions)
                    Shape params_prime_shape(params_shape);
                    params_prime_shape.erase(params_prime_shape.begin(),
                                             params_prime_shape.begin() + axis);
                    // prepare shape of indices_prime
                    size_t indices_ndim = static_cast<size_t>(indices_shape.size());
                    Shape indices_prime_shape;
                    // prepare shape of out_prime (same as params_prime except for first dim)
                    Shape out_prime_shape(params_prime_shape);
                    if (indices_ndim > 0)
                    {
                        out_prime_shape[0] = indices_shape[indices_ndim - 1];
                        indices_prime_shape.emplace_back(indices_shape[indices_ndim - 1]);
                    }
                    else
                    {
                        out_prime_shape[0] = 1;
                    }
                    indices_prime_shape.emplace_back(1);

                    // Create a CoordinateTransform for "out" that visits the outer "axis"
                    // dimensions
                    size_t out_ndim = static_cast<size_t>(out_shape.size());
                    Coordinate out_outer_start_corner(out_ndim, 0);
                    Coordinate out_outer_end_corner(out_shape);
                    for (size_t i = axis; i < out_ndim; i++)
                    {
                        out_outer_end_corner[i] = 1;
                    }
                    Strides out_outer_strides(out_ndim, 1);
                    AxisVector out_outer_axis_order(out_ndim);
                    std::iota(out_outer_axis_order.begin(), out_outer_axis_order.end(), 0);
                    CoordinateTransform out_outer_transform(out_shape,
                                                            out_outer_start_corner,
                                                            out_outer_end_corner,
                                                            out_outer_strides,
                                                            out_outer_axis_order);

                    // Create a CoordinateTransform for "params" that visits the outer "axis"
                    // dimensions
                    size_t params_ndim = static_cast<size_t>(params_shape.size());
                    Coordinate params_outer_start_corner(params_ndim, 0);
                    Coordinate params_outer_end_corner(params_shape);
                    for (size_t i = axis; i < params_ndim; i++)
                    {
                        params_outer_end_corner[i] = 1;
                    }
                    Strides params_outer_strides(params_ndim, 1);
                    AxisVector params_outer_axis_order(params_ndim);
                    std::iota(params_outer_axis_order.begin(), params_outer_axis_order.end(), 0);
                    CoordinateTransform params_outer_transform(params_shape,
                                                               params_outer_start_corner,
                                                               params_outer_end_corner,
                                                               params_outer_strides,
                                                               params_outer_axis_order);

                    // Create a CoordinateTransform for "indices" that visits only the first element
                    // along inner most axis
                    Coordinate indices_outer_start_corner(indices_ndim, 0);
                    Coordinate indices_outer_end_corner(indices_shape);
                    if (indices_ndim > 0)
                    {
                        indices_outer_end_corner[indices_ndim - 1] = 1;
                    }
                    Strides indices_outer_strides(indices_ndim, 1);
                    AxisVector indices_outer_axis_order(indices_ndim);
                    std::iota(indices_outer_axis_order.begin(), indices_outer_axis_order.end(), 0);
                    CoordinateTransform indices_outer_transform(indices_shape,
                                                                indices_outer_start_corner,
                                                                indices_outer_end_corner,
                                                                indices_outer_strides,
                                                                indices_outer_axis_order);

                    // Create an inner CoordinateTransfrom for "out"
                    size_t out_inner_ndim = out_ndim - axis;
                    Shape out_inner_shape(out_shape);
                    out_inner_shape.erase(out_inner_shape.begin(), out_inner_shape.begin() + axis);
                    Coordinate out_inner_start_corner(out_inner_ndim, 0);
                    Coordinate out_inner_end_corner(out_inner_shape);
                    if (indices_ndim > 0)
                    {
                        out_inner_end_corner[indices_ndim - 1] = 1;
                    }
                    for (size_t i = indices_ndim; i < out_inner_ndim; i++)
                    {
                        out_inner_end_corner[i] = 1;
                    }
                    Strides out_inner_strides(out_inner_ndim, 1);
                    AxisVector out_inner_axis_order(out_inner_ndim);
                    std::iota(out_inner_axis_order.begin(), out_inner_axis_order.end(), 0);
                    CoordinateTransform out_inner_transform(out_inner_shape,
                                                            out_inner_start_corner,
                                                            out_inner_end_corner,
                                                            out_inner_strides,
                                                            out_inner_axis_order);

                    auto out_outer_coord_iter = out_outer_transform.begin();
                    for (const Coordinate& params_outer_coord : params_outer_transform)
                    {
                        if (out_outer_coord_iter == out_outer_transform.end())
                            break;
                        const T* params_prime =
                            &params[params_outer_transform.index(params_outer_coord)];
                        T* out_outer = &out[out_outer_transform.index(*out_outer_coord_iter)];

                        auto out_inner_coord_iter = out_inner_transform.begin();
                        for (const Coordinate& indices_outer_coord : indices_outer_transform)
                        {
                            if (out_inner_coord_iter == out_inner_transform.end())
                                break;
                            const U* indices_prime =
                                &indices[indices_outer_transform.index(indices_outer_coord)];
                            T* out_prime =
                                &out_outer[out_inner_transform.index(*out_inner_coord_iter)];
                            gather_nd<T, U>(params_prime,
                                            indices_prime,
                                            out_prime,
                                            params_prime_shape,
                                            indices_prime_shape,
                                            out_prime_shape);
                            ++out_inner_coord_iter;
                        }
                        ++out_outer_coord_iter;
                    }
                }
            } // namespace v0

            namespace v1
            {
                namespace
                {
                    template <bool check>
                    using Required = typename std::enable_if<check, bool>::type;

                    template <typename It>
                    struct IsRandomAccessIt
                    {
                        static constexpr bool value =
                            std::is_same<typename It::iterator_category,
                                         std::random_access_iterator_tag>::value;
                    };

                    /// @brief Span should mimic std::span
                    template <typename Element>
                    class Span
                    {
                    public:
                        template <typename, typename = size_t>
                        struct is_complete : std::false_type
                        {
                        };

                        template <typename T>
                        struct is_complete<T, decltype(sizeof(T))> : std::true_type
                        {
                        };

                        static_assert(
                            std::is_object<Element>::value,
                            "Element must be an object type (not a reference type or void)");
                        static_assert(
                            is_complete<Element>::value,
                            "Element must be a complete type (not a forward declaration)");
                        static_assert(!std::is_abstract<Element>::value,
                                      "Element cannot be an abstract class type");

                        constexpr Span(const Element* data = nullptr, std::size_t size = 0)
                            : m_data{data}
                            , m_size{size}
                        {
                        }

                        using value_type = Element;
                        using size_type = std::size_t;

                        constexpr const Element* begin() const noexcept { return m_data; }
                        constexpr const Element* end() const noexcept { return m_data + m_size; }
                        friend constexpr const Element* begin(const Span& s) noexcept
                        {
                            return s.begin();
                        }
                        friend constexpr const Element* end(const Span& s) noexcept
                        {
                            return s.end();
                        }
                        constexpr std::size_t size() const noexcept { return m_size; }
                        constexpr bool empty() const noexcept { return !m_size; }
                        constexpr const Element& front() const noexcept { return *m_data; }
                        constexpr const Element& back() const noexcept
                        {
                            return *(m_data + (m_size - 1));
                        }
                        constexpr const Element& operator[](std::size_t idx) const
                        {
                            return *(m_data + idx);
                        }
                        const Element& at(std::size_t idx) const
                        {
                            if (idx >= m_size)
                            {
                                throw std::out_of_range{"Index: " + std::to_string(idx) +
                                                        " out of range"};
                            }
                            return *(m_data + idx);
                        }

                        Span subspan(std::size_t offset,
                                     std::size_t size = std::numeric_limits<std::size_t>::max())
                        {
                            if (offset > m_size)
                            {
                                return {};
                            }
                            return {m_data + offset, std::min(size, m_size - offset)};
                        }

                    private:
                        const Element* m_data;
                        std::size_t m_size;
                    };

                    template <typename Iterator, Required<IsRandomAccessIt<Iterator>::value> = true>
                    Span<typename Iterator::value_type> span(Iterator begin, Iterator end)
                    {
                        using Span = Span<typename Iterator::value_type>;
                        return Span{
                            std::addressof(*begin),
                            static_cast<typename Span::size_type>(std::distance(begin, end))};
                    }

                    template <typename... Args>
                    using void_t = void;
                    template <typename Container,
                              typename = void_t<decltype(std::declval<Container>().data()),
                                                decltype(std::declval<Container>().size())>>
                    Span<typename Container::value_type> span(const Container& c)
                    {
                        return Span<typename Container::value_type>{c.data(), c.size()};
                    }

                    template <typename Element>
                    Span<Element> span(const Element* data, std::size_t size)
                    {
                        return Span<Element>{data, size};
                    }

                    template <typename Container>
                    Shape to_shape(const Container& c)
                    {
                        return Shape(begin(c), end(c));
                    }

                    template <typename Container>
                    std::vector<size_t>
                        join(const Container& c1, const Container c2, const Container c3)
                    {
                        static_assert(std::is_same<typename Container::value_type, size_t>::value,
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

                    if (params_shape.size() < axis + 1)
                    {
                        throw std::domain_error{"Not enough axes in param_shape."};
                    }

                    const auto reminder_part_shape = span(params_shape).subspan(axis + 1);

                    const auto found_out_shape =
                        join(params_axes_part, span(indices_shape), reminder_part_shape);

                    if (found_out_shape != out_shape)
                    {
                        throw std::runtime_error{"calculation mists"};
                    }

                    const auto batch_shape = span(params_shape).subspan(axis);

                    const auto batch_size = shape_size(batch_shape);

                    const auto copy_size = shape_size(reminder_part_shape);

                    const size_t copy_round_in_batch =
                        indices_shape.empty() ? 1
                                              : shape_size(indices_shape) / indices_shape.back();
                    const size_t round_batch_offset =
                        indices_shape.empty() ? 1 : indices_shape.back();

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
                                        input_indices[ii] < 0
                                            ? batch_shape.front() + input_indices[ii]
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
            } // namespace v1

            using namespace v1;

        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
