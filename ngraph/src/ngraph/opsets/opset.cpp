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

#include "ngraph/opsets/opset.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ops.hpp"

std::mutex& ngraph::OpSet::get_mutex()
{
    static std::mutex opset_mutex;
    return opset_mutex;
}

ngraph::Node* ngraph::OpSet::create(const std::string& name) const
{
    auto type_info_it = m_name_type_info_map.find(name);
    if (type_info_it == m_name_type_info_map.end())
    {
        NGRAPH_WARN << "Couldn't create operator of type: " << name
                    << " . Operation not registered in opset.";
        return nullptr;
    }
    return m_factory_registry.create(type_info_it->second);
}

ngraph::Node* ngraph::OpSet::create_insensitive(const std::string& name) const
{
    auto type_info_it = m_case_insensitive_type_info_map.find(to_upper_name(name));
    return type_info_it == m_name_type_info_map.end()
               ? nullptr
               : m_factory_registry.create(type_info_it->second);
}

const ngraph::OpSet& ngraph::get_opset0()
{
    static std::mutex init_mutex;
    static OpSet opset;
    if (opset.size() == 0)
    {
        std::lock_guard<std::mutex> guard(init_mutex);
        if (opset.size() == 0)
        {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset0_tbl.hpp"
#undef NGRAPH_OP
        }
    }
    return opset;
}

const ngraph::OpSet& ngraph::get_opset1()
{
    static std::mutex init_mutex;
    static bool opset_is_initialized = false;
    static OpSet opset;
    if (!opset_is_initialized)
    {
        std::lock_guard<std::mutex> guard(init_mutex);
        if (!opset_is_initialized)
        {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset1_tbl.hpp"
#undef NGRAPH_OP
            opset_is_initialized = true;
        }
    }
    return opset;
}

const ngraph::OpSet& ngraph::get_opset2()
{
    static std::mutex init_mutex;
    static bool opset_is_initialized = false;
    static OpSet opset;
    if (!opset_is_initialized)
    {
        std::lock_guard<std::mutex> guard(init_mutex);
        if (!opset_is_initialized)
        {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset2_tbl.hpp"
#undef NGRAPH_OP
            opset_is_initialized = true;
        }
    }
    return opset;
}

const ngraph::OpSet& ngraph::get_opset3()
{
    static std::mutex init_mutex;
    static bool opset_is_initialized = false;
    static OpSet opset;
    if (!opset_is_initialized)
    {
        std::lock_guard<std::mutex> guard(init_mutex);
        if (!opset_is_initialized)
        {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset3_tbl.hpp"
#undef NGRAPH_OP
            opset_is_initialized = true;
        }
    }
    return opset;
}

const ngraph::OpSet& ngraph::get_opset4()
{
    static std::mutex init_mutex;
    static bool opset_is_initialized = false;
    static OpSet opset;
    if (!opset_is_initialized)
    {
        std::lock_guard<std::mutex> guard(init_mutex);
        if (!opset_is_initialized)
        {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset4_tbl.hpp"
#undef NGRAPH_OP
            opset_is_initialized = true;
        }
    }
    return opset;
}

const ngraph::OpSet& ngraph::get_ie_opset()
{
    static std::mutex init_mutex;
    static bool opset_is_initialized = false;
    static OpSet opset;
    if (!opset_is_initialized)
    {
        std::lock_guard<std::mutex> guard(init_mutex);
        if (!opset_is_initialized)
        {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset1_tbl.hpp"
#include "ngraph/opsets/opset2_tbl.hpp"
#include "ngraph/opsets/opset3_tbl.hpp"
#include "ngraph/opsets/opset4_tbl.hpp"
#undef NGRAPH_OP
            opset_is_initialized = true;
        }
    }
    return opset;
}
