// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/old_api_map_attribute.hpp"

using namespace ov;

bool OldApiMap::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("order", m_value.m_order);
    visitor.on_attribute("type", m_value.m_type);
    return true;
}