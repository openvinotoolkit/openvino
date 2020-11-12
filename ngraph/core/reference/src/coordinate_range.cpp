//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include "ngraph/coordinate_range.hpp"

#include "ngraph/coordinate_index.hpp"

namespace ngraph
{
    namespace coordinates
    {
        size_t SliceRange::index() const { return coordinate_index(m_coordinate, m_source_shape); }
        SliceRange::Iterator& SliceRange::Iterator::operator++()
        {
            if (m_r)
            {
                for (auto axis = m_r->m_coordinate.size(); axis-- > 0;)
                {
                    m_r->m_coordinate[axis] += m_r->m_strides[axis];
                    if (m_r->m_coordinate[axis] < m_r->m_bounds.upper[axis])
                    {
                        return *this;
                    }
                    m_r->m_coordinate[axis] = m_r->m_bounds.lower[axis];
                }
                m_r = nullptr;
            }
            return *this;
        }

        size_t ReverseRange::index() const
        {
            do_reverse();
            return coordinate_index(m_reversed_coordinate, m_source_shape);
        }

        ReverseRange::Iterator& ReverseRange::Iterator::operator++()
        {
            if (m_r)
            {
                for (auto axis = m_r->m_coordinate.size(); axis-- > 0;)
                {
                    ++m_r->m_coordinate[axis];
                    if (m_r->m_coordinate[axis] < m_r->m_source_shape[axis])
                    {
                        return *this;
                    }
                    m_r->m_coordinate[axis] = 0;
                }
                m_r = nullptr;
            }
            return *this;
        }

        void ReverseRange::do_reverse() const
        {
            m_reversed_coordinate = m_coordinate;

            for (const auto& i : m_reversed_axes)
            {
                m_reversed_coordinate[i] = m_source_shape[i] - m_reversed_coordinate[i] - 1;
            }
        }

    } // namespace coordinates
} // namespace ngraph
