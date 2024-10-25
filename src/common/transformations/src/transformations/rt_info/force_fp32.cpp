// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/force_fp32.hpp"

using namespace ov;

bool ForceFP32::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("value", value);
    return true;
}
