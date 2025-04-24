// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/old_api_map_element_type_attribute.hpp"

using namespace ov;

bool OldApiMapElementType::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("value", value);
    return true;
}
