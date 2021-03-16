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

#ifdef _WIN32
#else
#include <cxxabi.h>
#endif

#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

NGRAPH_RTTI_DEFINITION(ngraph::pass::FunctionPass, "ngraph::pass::FunctionPass", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::NodePass, "ngraph::pass::NodePass", 0);

pass::PassBase::PassBase()
    : m_property{all_pass_property_off}
    , m_pass_config(std::make_shared<PassConfig>())
{
}

bool pass::PassBase::get_property(const PassPropertyMask& prop) const
{
    return m_property.is_set(prop);
}

void pass::PassBase::set_property(const PassPropertyMask& prop, bool value)
{
    if (value)
    {
        m_property.set(prop);
    }
    else
    {
        m_property.clear(prop);
    }
}

std::string pass::PassBase::get_name() const
{
    if (m_name.empty())
    {
        const PassBase* p = this;
        std::string pass_name = typeid(*p).name();
#ifndef _WIN32
        int status;
        pass_name = abi::__cxa_demangle(pass_name.c_str(), nullptr, nullptr, &status);
#endif
        return pass_name;
    }
    else
    {
        return m_name;
    }
}

void pass::PassBase::set_callback(const param_callback& callback)
{
    m_pass_config->set_callback(callback);
}

// The symbols are requiered to be in cpp file to workaround RTTI issue on Android LLVM

pass::FunctionPass::~FunctionPass() {}

pass::NodePass::~NodePass() {}
