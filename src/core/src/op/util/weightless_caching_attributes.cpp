// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/rt_info/weightless_caching_attributes.hpp"

#include "openvino/op/util/op_types.hpp"

bool ov::WeightlessCacheAttribute::is_copyable() const {
    return false;
}

bool ov::WeightlessCacheAttribute::is_copyable(const std::shared_ptr<ov::Node>& from,
                                               const std::shared_ptr<ov::Node>& to) const {
    if (!ov::op::util::is_constant(from) || !ov::op::util::is_constant(to)) {
        return false;
    }

    return from->get_element_type() != to->get_element_type();
}
