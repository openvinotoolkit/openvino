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

#pragma once

#include <map>
#include <string>

#include <ngraph/ngraph_visibility.hpp>

namespace ngraph
{
    namespace pass
    {
        class PassConfig;
    }
}

class NGRAPH_API ngraph::pass::PassConfig
{
public:
    PassConfig();
    const std::map<std::string, bool>& get_enables() const { return m_pass_enables; }
    void set_pass_enable(const std::string& name, bool enable);
    bool get_pass_enable(const std::string& name) const;
    const std::map<std::string, bool>& get_pass_attributes() const { return m_pass_attributes; }
    void set_pass_attribute(const std::string& name, bool enable);
    bool get_pass_attribute(const std::string& name) const;

private:
    std::map<std::string, bool> m_pass_enables;
    std::map<std::string, bool> m_pass_attributes;
};
