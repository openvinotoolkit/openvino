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

#include "ndimiterator.hpp"

namespace ngraph
{
    namespace runtime
    {
        class NDimRange{
        public:
            NDimRange() = default;
            NDimRange(const NDimRange& rhs) = default;
            NDimRange(NDimRange&& rhs) = default;
            NDimRange& operator = (const NDimRange& rhs) = default;
            NDimRange& operator = (NDimRange&& rhs) = default;

            virtual ~NDimRange() = default;

            NDimRange(const NDimIterator& first, const NDimIterator& last) :
                m_first{first}, m_last{last} {}

            NDimRange(const NDimIndex& fst_index, const NDimIndex& snd_index) :
                m_first{fst_index}, m_last{snd_index.next()} {}

            explicit NDimRange(const NDimIndex& snd_index) :
                m_first{snd_index.zeros()}, m_last{snd_index.next()} {}

            NDimIterator begin() const
            {
                return m_first;
            }

            NDimIterator end() const
            {
                return m_last;
            }
        private:
            NDimIterator m_first;
            NDimIterator m_last;
        };
    }
}
