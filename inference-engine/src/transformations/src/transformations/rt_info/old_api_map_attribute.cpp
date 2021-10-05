// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/old_api_map_attribute.hpp"

using namespace ov;

bool OldApiMap::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("order", m_value.m_order);
    visitor.on_attribute("element_type", m_value.m_legacy_type);
    return true;
}

bool ov::has_old_api_map(const Node * node) {
    const auto& rt_map = node->get_rt_info();
    return rt_map.count(OldApiMap::get_type_info_static());
}

OldApiMap ov::get_old_api_map(const Node * node) {
    const auto& rt_map = node->get_rt_info();
    const auto& var = rt_map.at(OldApiMap::get_type_info_static());
    return ngraph::as_type_ptr<OldApiMap>(var)->get();
}

void ov::set_old_api_map(Node * node, const OldApiMap& old_api_map) {
    auto& rt_map = node->get_rt_info();
    rt_map[OldApiMap::get_type_info_static()] = std::make_shared<OldApiMap>(old_api_map);
}
