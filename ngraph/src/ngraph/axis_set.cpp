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

#include "ngraph/axis_set.hpp"
#include "ngraph/util.hpp"

ngraph::AxisSet::AxisSet()
    : std::set<size_t>()
{
}

ngraph::AxisSet::AxisSet(const std::initializer_list<size_t>& axes)
    : std::set<size_t>(axes)
{
}

ngraph::AxisSet::AxisSet(const std::set<size_t>& axes)
    : std::set<size_t>(axes)
{
}

ngraph::AxisSet::AxisSet(const std::vector<size_t>& axes)
    : std::set<size_t>(axes.begin(), axes.end())
{
}

ngraph::AxisSet::AxisSet(const AxisSet& axes)
    : std::set<size_t>(axes)
{
}

ngraph::AxisSet& ngraph::AxisSet::operator=(const AxisSet& v)
{
    static_cast<std::set<size_t>*>(this)->operator=(v);
    return *this;
}

ngraph::AxisSet& ngraph::AxisSet::operator=(AxisSet&& v) noexcept
{
    static_cast<std::set<size_t>*>(this)->operator=(v);
    return *this;
}

std::vector<int64_t> ngraph::AxisSet::to_vector() const
{
    return std::vector<int64_t>(this->begin(), this->end());
}

std::ostream& ngraph::operator<<(std::ostream& s, const AxisSet& axis_set)
{
    s << "AxisSet{";
    s << ngraph::join(axis_set);
    s << "}";
    return s;
}

const std::vector<int64_t>& ngraph::AttributeAdapter<ngraph::AxisSet>::get()
{
    if (!m_buffer_valid)
    {
        m_buffer.clear();
        for (auto elt : m_ref)
        {
            m_buffer.push_back(elt);
        }
        m_buffer_valid = true;
    }
    return m_buffer;
}

void ngraph::AttributeAdapter<ngraph::AxisSet>::set(const std::vector<int64_t>& value)
{
    m_ref = AxisSet();
    for (auto elt : value)
    {
        m_ref.insert(elt);
    }
    m_buffer_valid = false;
}

constexpr ngraph::DiscreteTypeInfo ngraph::AttributeAdapter<ngraph::AxisSet>::type_info;
