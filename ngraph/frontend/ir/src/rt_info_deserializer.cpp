// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ir_frontend/utility.hpp>
#include <pugixml.hpp>
#include <rt_info_deserializer.hpp>
#include <transformations/rt_info/attributes.hpp>

using namespace ov;

void RTInfoDeserializer::on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) {
    std::string val;
    if (!getStrAttribute(m_node, name, val))
        return;
    if (auto a = ngraph::as_type<ngraph::AttributeAdapter<std::set<std::string>>>(&adapter)) {
        std::set<std::string> ss;
        str_to_container(val, ss);
        a->set(ss);
    } else {
        IR_THROW("Not implemented");
    }
}