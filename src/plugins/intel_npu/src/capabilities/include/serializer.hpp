//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

#include "meta.hpp"

namespace compat::tlv {

size_t alignment();

class Serializer {
public:
    template <class T>
    auto& append(const T& object) {
        static_assert(compat::meta::IsCapability<T>::value, "Type does not satisfy Capability concept");
        // it is safe and intentional to do reinterpret_cast here
        // as Capability classes binary representation is kept cross-compatible
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        const auto begin = reinterpret_cast<const uint8_t*>(&object);
        appendImpl(T::ID, T::POLICY, begin, alignof(T), sizeof(T));
        return *this;
    }

    template <class T>
    auto& append() {
        static_assert(compat::meta::IsCapability<T>::value, "Type does not satisfy Capability concept");
        static_assert(!compat::meta::HasMemberFunctionCheck<T>::value, "Capability is serialized without state, but it has non-static isCompatible check");
        appendImpl(T::ID, T::POLICY);
        return *this;
    }

    std::vector<uint8_t> done();

private:
    void appendImpl(uint16_t id, bool policy);
    void appendImpl(uint16_t id, bool policy, const uint8_t* ptr, size_t alignment, size_t size);
    std::vector<uint8_t> _buffer;
};

}  // namespace compat::tlv
