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

#pragma once

#include "ndimindex.hpp"

namespace ngraph
{
    namespace runtime
    {
        class NDimIterator{
        public:
            NDimIterator() = default;
            NDimIterator(const NDimIterator& rhs) = default;
            NDimIterator(NDimIterator&& rhs) = default;
            NDimIterator& operator = (const NDimIterator& rhs) = default;
            NDimIterator& operator = (NDimIterator&& rhs) = default;

            virtual ~NDimIterator() = default;

            explicit NDimIterator(const NDimIndex& index) : m_index{index} {}

            NDimIterator& operator ++()
            {
                ++m_index;
                return *this;
            }

            NDimIterator operator ++ (int)
            {
                NDimIterator old_value{*this};
                ++(*this);
                return old_value;
            }

            bool operator == (const NDimIterator& rhs) const
            {
                return m_index == rhs.m_index;
            }

            bool operator != (const NDimIterator& rhs) const
            {
                return m_index != rhs.m_index;
            }

            NDimIndex operator * () const
            {
                return m_index;
            }
        private:
            NDimIndex m_index;
        };
    }
}
