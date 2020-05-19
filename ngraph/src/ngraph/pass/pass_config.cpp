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
#include "ngraph/env_util.hpp"
#include "ngraph/except.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

// TODO: Add file-based configuration support
pass::PassConfig::PassConfig()
{
    //
    // Parses the semi-colon separated environment string passed through NGRAPH_PASS_ENABLES
    // and returns the pass names and whether they should be enabled or disabled in the
    // provided unordered_map. Implementation of pass selection is up to the backend
    // E.g., NGRAPH_PASS_ENABLES="CoreFusion:0;LikeReplacement:1;CPUCollapseDims" would
    //       set disables on CoreFusion and enables on LikeReplacement and CPUCollapseDims
    //
    string pass_enables = getenv_string("NGRAPH_PASS_ENABLES");
    if (!pass_enables.empty())
    {
        stringstream ss;
        ss << pass_enables;
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ';');
            auto split_str = split(substr, ':', false);
            switch (split_str.size())
            {
            case 1: m_pass_enables.emplace(split_str[0], true); break;
            case 2: m_pass_enables.emplace(split_str[0], parse_string<bool>(split_str[1])); break;
            default: throw ngraph_error("Unexpected string in NGRAPH_PASS_ENABLES: " + substr);
            }
        }
    }
    //
    //  Parses the semi-colon separated environment string passed through NGRAPH_PASS_ATTRIBUTES
    //  and returns the pass attributes and whether they should be enabled or disabled in the
    //  provided unordered_map. Naming of pass attributes is up to the backends.
    //
    //  For example:
    //  NGRAPH_PASS_ATTRIBUTES="OptimizeForMemory=0;MemoryAssignment::ReuseMemory=1;UseDefaultLayouts"
    //  would set false on "OptimizeForMemory", true on "MemoryAssignment::ReuseMemory" and true on
    //  "UseDefaultLayouts"
    //
    static const string pass_attributes = getenv_string("NGRAPH_PASS_ATTRIBUTES");
    if (!pass_attributes.empty())
    {
        stringstream ss;
        ss << pass_attributes;
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ';');
            auto split_str = split(substr, '=', false);
            switch (split_str.size())
            {
            case 1: m_pass_attributes.emplace(split_str[0], true); break;
            case 2:
                m_pass_attributes.emplace(split_str[0], parse_string<bool>(split_str[1]));
                break;
            default: throw ngraph_error("Unexpected string in NGRAPH_PASS_ATTRIBUTES: " + substr);
            }
        }
    }
}

void pass::PassConfig::set_pass_enable(const string& name, bool enable)
{
    m_pass_enables[name] = enable;
}

bool pass::PassConfig::get_pass_enable(const string& name) const
{
    auto it = m_pass_enables.find(name);
    if (it != m_pass_enables.end())
    {
        return it->second;
    }
    return false;
}

void pass::PassConfig::set_pass_attribute(const string& name, bool enable)
{
    m_pass_attributes[name] = enable;
}

bool pass::PassConfig::get_pass_attribute(const string& name) const
{
    auto it = m_pass_attributes.find(name);
    if (it != m_pass_attributes.end())
    {
        return it->second;
    }
    return false;
}
