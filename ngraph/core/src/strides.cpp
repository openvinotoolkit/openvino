//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/strides.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& s, const Strides& strides)
{
    s << "Strides{";
    s << ngraph::join(strides);
    s << "}";
    return s;
}

ngraph::Strides::Strides()
    : std::vector<size_t>()
{
}

ngraph::Strides::Strides(const std::initializer_list<size_t>& axis_strides)
    : std::vector<size_t>(axis_strides)
{
}

ngraph::Strides::Strides(const std::vector<size_t>& axis_strides)
    : std::vector<size_t>(axis_strides)
{
}

ngraph::Strides::Strides(const Strides& axis_strides)
    : std::vector<size_t>(axis_strides)
{
}

ngraph::Strides::Strides(size_t n, size_t initial_value)
    : std::vector<size_t>(n, initial_value)
{
}

ngraph::Strides& ngraph::Strides::operator=(const Strides& v)
{
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ngraph::Strides& ngraph::Strides::operator=(Strides&& v) noexcept
{
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

constexpr DiscreteTypeInfo AttributeAdapter<Strides>::type_info;
