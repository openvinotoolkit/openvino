// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    return type_info_it == m_case_insensitive_type_info_map.end()
               ? nullptr
               : m_factory_registry.create(type_info_it->second);
}

const ngraph::OpSet& ngraph::get_opset1()
{
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset1_tbl.hpp"
#undef NGRAPH_OP
    });
    return opset;
}

const ngraph::OpSet& ngraph::get_opset2()
{
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset2_tbl.hpp"
#undef NGRAPH_OP
    });
    return opset;
}

const ngraph::OpSet& ngraph::get_opset3()
{
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset3_tbl.hpp"
#undef NGRAPH_OP
    });
    return opset;
}

const ngraph::OpSet& ngraph::get_opset4()
{
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset4_tbl.hpp"
#undef NGRAPH_OP
    });
    return opset;
}

const ngraph::OpSet& ngraph::get_opset5()
{
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset5_tbl.hpp"
#undef NGRAPH_OP
    });
    return opset;
}

const ngraph::OpSet& ngraph::get_opset6()
{
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset6_tbl.hpp"
#undef NGRAPH_OP
    });
    return opset;
}

const ngraph::OpSet& ngraph::get_opset7()
{
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset7_tbl.hpp"
#undef NGRAPH_OP
    });
    return opset;
}

const ngraph::OpSet& ngraph::get_opset8()
{
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset8_tbl.hpp"
#undef NGRAPH_OP
    });
    return opset;
}
