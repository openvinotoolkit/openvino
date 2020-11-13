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
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace details
    {
        class CoordinateImpl
        {
        public:
            enum
            {
                max_size = 20
            };

            CoordinateImpl() = default;

            template <typename It>
            CoordinateImpl(It first, It last)
            {
                std::copy(first, last, m_data.data());
                m_size = std::distance(first, last);
            }

            template <typename T>
            CoordinateImpl(const T& v)
                : CoordinateImpl(std::begin(v), std::end(v))
            {
            }

            CoordinateImpl(size_t s, size_t v = 0)
                : m_size{s}
            {
                if (v != 0)
                {
                    std::fill_n(data(), s, v);
                }
                else
                {
                    m_data = {};
                }
            }

            size_t operator[](size_t i) const noexcept { return m_data[i]; }
            size_t& operator[](size_t i) noexcept { return m_data[i]; }

            size_t at(size_t i) const
            {
                if (i >= m_size)
                {
                    throw std::runtime_error{"index out of range"};
                }
                return m_data[i];
            }

            size_t& at(size_t i)
            {
                if (i >= m_size)
                {
                    throw std::runtime_error{"index out of range"};
                }
                return m_data[i];
            }

            void push_back(size_t v)
            {
                m_data[m_size] = v;
                ++m_size;
            }

            size_t size() const noexcept { return m_size; }

            size_t* data() noexcept { return m_data.data(); }

            const size_t* data() const noexcept { return m_data.data(); }

            const size_t* begin() const { return data(); }
            size_t* begin() { return data(); }
            const size_t* end() const { return std::next(data(), size()); }
            size_t* end() { return std::next(data(), size()); }

            friend bool operator<(const CoordinateImpl& lhs, const CoordinateImpl& rhs)
            {
                return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
            }

            friend bool operator==(const CoordinateImpl& lhs, const CoordinateImpl& rhs)
            {
                return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
            }
            friend bool operator!=(const CoordinateImpl& lhs, const CoordinateImpl& rhs)
            {
                return !(lhs == rhs);
            }

        private:
            std::array<size_t, max_size> m_data;
            size_t m_size = 0;
        };
    } // namespace details
    /// \brief Coordinates for a tensor element
    class Coordinate : public details::CoordinateImpl
    {
    public:
        NGRAPH_API Coordinate();
        NGRAPH_API Coordinate(const std::initializer_list<size_t>& axes);

        NGRAPH_API Coordinate(const Shape& shape);

        NGRAPH_API Coordinate(const std::vector<size_t>& axes);

        NGRAPH_API Coordinate(const Coordinate& axes);

        NGRAPH_API Coordinate(size_t n, size_t initial_value = 0);

        NGRAPH_API ~Coordinate();

        template <class InputIterator>
        Coordinate(InputIterator first, InputIterator last)
            : details::CoordinateImpl(first, last)
        {
        }

        operator std::vector<size_t>() const
        {
            return std::vector<size_t>(data(), std::next(data(), size()));
        }

        NGRAPH_API Coordinate& operator=(const Coordinate& v);

        NGRAPH_API Coordinate& operator=(Coordinate&& v) noexcept;
    };

    template <>
    class NGRAPH_API AttributeAdapter<Coordinate>
        : public IndirectVectorValueAccessor<Coordinate, std::vector<int64_t>>
    {
    public:
        AttributeAdapter(Coordinate& value)
            : IndirectVectorValueAccessor<Coordinate, std::vector<int64_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<Coordinate>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const Coordinate& coordinate);
} // namespace ngraph
