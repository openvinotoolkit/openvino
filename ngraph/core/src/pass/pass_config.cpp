// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
