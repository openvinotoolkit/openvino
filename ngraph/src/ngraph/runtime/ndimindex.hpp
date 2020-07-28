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

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <tuple>
#include <vector>

namespace ngraph
{
    namespace runtime
    {
        class NDimIndex
        {
        public:
            NDimIndex() = default;
            NDimIndex(const NDimIndex& rhs) = default;
            NDimIndex(NDimIndex&& rhs) = default;
            NDimIndex& operator=(const NDimIndex& rhs) = default;
            NDimIndex& operator=(NDimIndex&& rhs) = default;

            virtual ~NDimIndex() = default;

            NDimIndex(const std::vector<int64_t>& value,
                      const std::vector<int64_t>& low_limit,
                      const std::vector<int64_t>& high_limit)
                : m_value{value}
                , m_low_limit{low_limit}
                , m_high_limit{high_limit}
            {
            }

            NDimIndex(const std::vector<std::int64_t>& value,
                      const std::vector<int64_t>& high_limit)
                : m_value{value}
                , m_low_limit{std::vector<std::int64_t>(value.size(), 0)}
                , m_high_limit{high_limit}
            {
            }

            bool operator==(const NDimIndex& rhs) const;
            bool operator!=(const NDimIndex& rhs) const;

            NDimIndex& operator++();
            NDimIndex operator++(int);

            NDimIndex next() const;

            int64_t& operator[](std::size_t idx) { return m_value[idx]; }
            int64_t operator[](std::size_t idx) const { return m_value[idx]; }
            std::vector<std::int64_t> get_low_limit() const { return m_low_limit; }
            std::vector<std::int64_t> get_high_limit() const { return m_high_limit; }
            std::size_t size() const { return m_value.size(); }
            NDimIndex zeros() const
            {
                return {std::vector<std::int64_t>(m_value.size(), 0), m_low_limit, m_high_limit};
            }

            NDimIndex after_high_limit() const;

            friend std::ostream& operator<<(std::ostream& ostr, const NDimIndex& index);

        private:
            std::vector<std::int64_t> m_value;
            std::vector<std::int64_t> m_low_limit;
            std::vector<std::int64_t> m_high_limit;
        };

        std::ostream& operator<<(std::ostream& ostr, const NDimIndex& index);
    }
}
