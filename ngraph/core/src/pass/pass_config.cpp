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

#include "ngraph/pass/pass_config.hpp"

using namespace ngraph;

pass::param_callback pass::PassConfig::get_callback(const DiscreteTypeInfo& type_info) const
{
    const auto& it = m_callback_map.find(type_info);
    if (it != m_callback_map.end())
    {
        return it->second;
    }
    else
    {
        return m_callback;
    }
}

void pass::PassConfig::enable(const ngraph::DiscreteTypeInfo& type_info)
{
    m_disabled.erase(type_info);
    m_enabled.insert(type_info);
}

void pass::PassConfig::disable(const ngraph::DiscreteTypeInfo& type_info)
{
    m_enabled.erase(type_info);
    m_disabled.insert(type_info);
}

void pass::PassConfig::add_disabled_passes(const PassConfig& rhs)
{
    for (const auto& pass : rhs.m_disabled)
    {
        if (is_enabled(pass))
            continue;
        disable(pass);
    }
}
