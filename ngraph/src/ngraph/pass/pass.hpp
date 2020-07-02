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

#ifdef _WIN32
#else
#include <cxxabi.h>
#endif

#include <list>
#include <memory>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/manager_state.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace pass
    {
        class PassBase;
        class ModulePass;
        class FunctionPass;
        class NodePass;
        class CallGraphPass;
        class Manager;
        enum class FusionType : uint32_t
        {
            //`DIFFERENTIABLE_FUSIONS` produce ops that support autodiff
            // i.e. implement `generate_adjoints`
            DIFFERENTIABLE_FUSIONS = 0x1,
            REGULAR_FUSIONS = 0x2,
            //`FOP_FUSIONS` produce ops in the FusedOps category that might
            // not be supported by all backends
            FOP_FUSIONS = 0x4,
            ALL_FUSIONS = 0xFFFFFFFF
        };
        typedef EnumMask<FusionType> FusionTypeMask;

        enum class PassProperty : uint32_t
        {
            // Pass requires node shapes to be static
            REQUIRE_STATIC_SHAPE = 0x1,
            // Pass transformation will change the function's dynamic state
            CHANGE_DYNAMIC_STATE = 1 << 1,
        };

        using param_callback = std::function<bool(const std::shared_ptr<const ::ngraph::Node>)>;
    }
}

template class NGRAPH_API ngraph::EnumMask<ngraph::pass::PassProperty>;

namespace ngraph
{
    namespace pass
    {
        typedef EnumMask<PassProperty> PassPropertyMask;
        const PassPropertyMask all_pass_property_off;
    }
}

class NGRAPH_API ngraph::pass::PassBase
{
    friend class Manager;

public:
    PassBase();
    virtual ~PassBase() {}
    /// Check if this pass has all the pass properties.
    bool get_property(const PassPropertyMask& prop_mask) const;

    void set_name(const std::string& name) { m_name = name; }
    std::string get_name() const
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

    void set_callback(const param_callback& callback)
    {
        m_transformation_callback = callback;
        m_has_default_callback = false;
    }

protected:
    ManagerState& get_state();
    void set_state(ManagerState&);
    void set_property(const PassPropertyMask& prop, bool value);

    param_callback m_transformation_callback = [](const std::shared_ptr<const Node>&) -> bool {
        return false;
    };
    bool m_has_default_callback = true;

private:
    PassPropertyMask m_property;
    ManagerState* m_state{nullptr};
    std::string m_name;
};

class NGRAPH_API ngraph::pass::ModulePass : public PassBase
{
public:
    virtual ~ModulePass();
    virtual bool run_on_module(std::vector<std::shared_ptr<ngraph::Function>>&) = 0;
};

class NGRAPH_API ngraph::pass::FunctionPass : public PassBase
{
public:
    virtual ~FunctionPass();
    virtual bool run_on_function(std::shared_ptr<ngraph::Function>) = 0;
};

class NGRAPH_API ngraph::pass::NodePass : public PassBase
{
public:
    virtual ~NodePass();
    virtual bool run_on_node(std::shared_ptr<ngraph::Node>) = 0;
};

class NGRAPH_API ngraph::pass::CallGraphPass : public PassBase
{
public:
    virtual ~CallGraphPass();
    virtual bool run_on_call_graph(const std::list<std::shared_ptr<ngraph::Node>>&) = 0;
    virtual bool run_on_call_graph(const std::vector<std::shared_ptr<ngraph::Node>>&);
};
