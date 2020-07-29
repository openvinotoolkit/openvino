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

#ifdef _WIN32
#else
#include <cxxabi.h>
#endif

#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"

using namespace std;
using namespace ngraph;

pass::PassBase::PassBase()
    : m_property{all_pass_property_off}
{
}

pass::ManagerState& pass::PassBase::get_state()
{
    return *m_state;
}

void pass::PassBase::set_state(ManagerState& state)
{
    m_state = &state;
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
    m_transformation_callback = callback;
    m_has_default_callback = false;
}

// The symbols are requiered to be in cpp file to workaround RTTI issue on Android LLVM

pass::ModulePass::~ModulePass()
{
}

pass::FunctionPass::~FunctionPass()
{
}

pass::NodePass::~NodePass()
{
}

pass::CallGraphPass::~CallGraphPass()
{
}

bool pass::CallGraphPass::run_on_call_graph(const std::vector<std::shared_ptr<ngraph::Node>>& nodes)
{
    list<shared_ptr<Node>> node_list;
    for (auto op : nodes)
    {
        node_list.push_back(op);
    }
    return run_on_call_graph(node_list);
}
