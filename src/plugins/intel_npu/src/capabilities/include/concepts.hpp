//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <stdio.h>
#include <cassert>
#include <type_traits>

#include "meta.hpp"

namespace compat {
namespace concepts {

struct Requirement {};
static_assert(std::is_empty<Requirement>::value, "Requirement is just a named void*");

struct Capability {
    virtual ~Capability() = default;

    Capability() = default;
    Capability(const Capability&) = delete;
    Capability(Capability&&) = delete;
    Capability& operator=(const Capability&) = delete;
    Capability& operator=(Capability&&) = delete;

    virtual bool isSatisfied(const Requirement*) const = 0;
};

template <class T>
class CapabilityModel : public Capability {
    static_assert(meta::IsCapability<T>::value, "Type does not satisfy Capability concept");

public:
    explicit CapabilityModel(const T* self) : _self(self) {
        // assert(_self != nullptr);
    }

    bool isSatisfied(const Requirement* requirement) const override {
        printf("!!!     CapabilityModel: self = %p !!!\n", _self);
        printf("!!!     CapabilityModel: requ = %p !!!\n", requirement);
        if constexpr (std::is_empty<T>::value) {
            // TODO: it's unexpected, throw?
            return true;
        } else {
            if constexpr (meta::HasStaticMemberFunctionCheck<T>::value) {
                return T::isCompatible(*reinterpret_cast<const T*>(requirement));
            } else {
                return _self->isCompatible(*reinterpret_cast<const T*>(requirement));
            }
            // return _self->isCompatible(*reinterpret_cast<const T*>(requirement));
        }

        return false;
    }

private:
    const T* _self;
};

}  // namespace concepts
}  // namespace compat
