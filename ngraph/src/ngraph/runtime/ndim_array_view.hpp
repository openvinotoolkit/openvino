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

#include <cstdint>
#include <cstddef>
#include "ndimindex.hpp"

namespace ngraph
{
    namespace runtime
    {
        template<typename T>
        class NDimArrayView{
        public:
            NDimArrayView() = default;
            NDimArrayView(const NDimArrayView& rhs) = default;
            NDimArrayView& operator = (const NDimArrayView& rhs) = default;

            virtual ~NDimArrayView() = default;

            explicit NDimArrayView(T* data) : m_data{data} {}

            T& operator [](const NDimIndex& index)
            {
                std::int64_t offset = get_offset(index);
                return m_data[offset];
            }

            T operator [](const NDimIndex& index) const
            {
                std::int64_t offset = get_offset(index);
                return m_data[offset];
            }
        private:
            T* m_data = nullptr;

            std::int64_t get_offset(const NDimIndex& index)
            {
                auto high_limit = index.get_high_limit();
                auto low_limit = index.get_low_limit();
                std::size_t len = index.size();
                std::int64_t offset = 0;
                for (std::size_t i = 0; i < len; ++i)
                {
                    offset = offset * (high_limit[i] - low_limit[i] + 1) + index[i] - low_limit[i];
                }
                return offset;
            }
        };
    }
}
