// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>
#include <utility>

#include "openvino/runtime/properties.hpp"

namespace intel_npu {

struct PropertyDescriptor final {
    bool isPublic;
    ov::PropertyMutability mutability;
    std::function<bool(const ov::AnyMap&)> isSupported;
    std::function<ov::Any(const ov::AnyMap&)> get;
    std::function<void(const ov::Any&)> set;
};

class PropertyRegistrationBase {
protected:
    /**
     * @brief Register a property with explicit support, getter, and setter callbacks.
     *
     * Registers the descriptor unconditionally. The property is exposed only when `isSupported` returns true. The
     * getter receives query-time arguments, and the setter receives the value supplied by the caller. Callers can use
     * these callbacks for config-backed properties or custom property implementations.
     */
    void register_property(std::string propertyName,
                           bool isPublic,
                           ov::PropertyMutability mutability,
                           std::function<bool(const ov::AnyMap&)> isSupported,
                           std::function<ov::Any(const ov::AnyMap&)> getter,
                           std::function<void(const ov::Any&)> setter) {
        _properties.emplace(
            propertyName,
            PropertyDescriptor{isPublic, mutability, std::move(isSupported), std::move(getter), std::move(setter)});
    }

    std::map<std::string, PropertyDescriptor> _properties;
};

}  // namespace intel_npu
