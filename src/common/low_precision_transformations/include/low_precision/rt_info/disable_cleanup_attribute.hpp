// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "low_precision/lpt_visibility.hpp"

namespace ov {

class LP_TRANSFORMATIONS_API DisableCleanupAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("LowPrecision::DisableCleanup", "", ov::RuntimeAttribute);
    DisableCleanupAttribute() = default;

    static ov::Any create(const std::shared_ptr<ov::Node>& node) {
        auto& rt = node->get_rt_info();
        return (rt[DisableCleanupAttribute::get_type_info_static()] = DisableCleanupAttribute());
    }

    bool is_copyable() const override {
        return false;
    }
};
} // namespace ov
