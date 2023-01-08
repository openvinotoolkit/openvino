// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pugixml.hpp>
#include <rt_info_deserializer.hpp>
#include <transformations/rt_info/attributes.hpp>

#include "openvino/frontend/exception.hpp"

using namespace ov;

void RTInfoDeserializer::on_adapter(const std::string& name, ValueAccessor<void>& adapter) {
    check_attribute_name(name);
    std::string val;
    if (!getStrAttribute(m_node, name, val))
        return;
    if (auto a = as_type<AttributeAdapter<std::set<std::string>>>(&adapter)) {
        std::set<std::string> ss;
        str_to_set_of_strings(val, ss);
        a->set(ss);
    } else {
        IE_THROW() << "Not implemented";
    }
}
