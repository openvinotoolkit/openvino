// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <utility>

#include "openvino/runtime/properties.hpp"

namespace intel_npu {

class Config;

struct PropertyDescriptor final {
    bool isPublic;
    ov::PropertyMutability mutability;
    std::function<ov::Any(const Config&)> get;
};

inline PropertyDescriptor make_property_descriptor(bool isPublic,
                                                   ov::PropertyMutability mutability,
                                                   std::function<ov::Any(const Config&)> get) {
    return PropertyDescriptor{isPublic, mutability, std::move(get)};
}

}  // namespace intel_npu