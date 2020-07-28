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

#include "ndimindex.hpp"

using namespace ngraph;

bool runtime::NDimIndex::operator==(const NDimIndex& rhs) const
{
    return std::tie(m_value, m_low_limit, m_high_limit) ==
           std::tie(rhs.m_value, rhs.m_low_limit, rhs.m_high_limit);
}

bool runtime::NDimIndex::operator!=(const NDimIndex& rhs) const
{
    return std::tie(m_value, m_low_limit, m_high_limit) !=
           std::tie(rhs.m_value, rhs.m_low_limit, rhs.m_high_limit);
}

NDimIndex& runtime::NDimIndex::operator++()
{
    int64_t carry = 1;
    auto high_limit_it = m_high_limit.rbegin();
    auto low_limit_it = m_low_limit.rbegin();

    for (auto value_it = m_value.rbegin(); value_it != m_value.rend(); ++value_it)
    {
        int64_t i = *value_it + carry;
        if (i > *high_limit_it)
        {
            *value_it = *low_limit_it;
        }
        else
        {
            *value_it = i;
            return *this;
        }
        ++high_limit_it;
        ++low_limit_it;
    }
    m_value[0] = m_high_limit[0] + 1;
    return *this;
}

NDimIndex runtime::NDimIndex::operator++(int)
{
    NDimIndex old_value{*this};
    ++(*this);
    return old_value;
}

NDimIndex runtime::NDimIndex::next() const
{
    NDimIndex temp{*this};
    ++temp;
    return temp;
}

std::ostream& runtime::operator<<(std::ostream& ostr, const NDimIndex& index)
{
    std::string value_str;
    std::string low_limit_str;
    std::string high_limit_str;

    std::size_t len = index.m_value.size();
    for (std::size_t i = 0; i < len; ++i)
    {
        value_str += std::to_string(index.m_value[i]) + ", ";
        low_limit_str += std::to_string(index.m_low_limit[i]) + ", ";
        high_limit_str += std::to_string(index.m_high_limit[i]) + ", ";
    }
    value_str.pop_back();
    value_str.pop_back();
    low_limit_str.pop_back();
    low_limit_str.pop_back();
    high_limit_str.pop_back();
    high_limit_str.pop_back();

    std::string resulting_str = "NDimensionalIndex{value: {" + value_str + "}, low_limit: {" +
                                low_limit_str + "}, high_limit: {" + high_limit_str + "}}";
    ostr << resulting_str;
    return ostr;
}

NDimIndex runtime::NDimIndex::after_high_limit() const
{
    NDimIndex temp{m_high_limit, m_low_limit, m_high_limit};
    ++temp;
    return temp;
}
