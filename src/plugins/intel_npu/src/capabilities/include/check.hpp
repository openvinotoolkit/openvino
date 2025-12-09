//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <type_traits>

#include "meta.hpp"
#include "concepts.hpp"
#include "tlv.hpp"

namespace compat {

struct HeaderHash {
    std::size_t operator()(const tlv::Header& header) const noexcept {
        return std::hash<uint16_t>{}(header.type);
    }
};

struct HeaderEqual {
    bool operator()(const tlv::Header& lhs, const tlv::Header& rhs) const noexcept {
        return lhs.type == rhs.type;
    }
};

using Capabilities = std::unordered_map<tlv::Header, std::unique_ptr<const concepts::Capability>, HeaderHash, HeaderEqual>;
using Requirements = std::unordered_map<tlv::Header, const concepts::Requirement*, HeaderHash, HeaderEqual>;

bool isCompatible(const Capabilities& capabilities, const Requirements& requirements);

struct ByteArrayView {
    template <class T>
    explicit ByteArrayView(const T& object)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): reinterpret_cast is by class design
        : ByteArrayView(reinterpret_cast<const uint8_t*>(&object), sizeof(object)) {
        static_assert(meta::IsMemCopyable<typename std::decay<T>::type>::value, "ByteArrayView supports only trivially copyable type of standard layout");
    }

    template <class T>
    /* implicit */ ByteArrayView(const std::vector<T>& buffer)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): reinterpret_cast is by class design
        : ByteArrayView(reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size()) {
        static_assert(meta::IsMemCopyable<T>::value, "ByteArrayView supports only trivially copyable type of standard layout");
    }

    ByteArrayView(const uint8_t* ptr, size_t size);

    const uint8_t* ptr;
    size_t size;
};

namespace tlv {

Requirements parseRequirements(ByteArrayView section);
Capabilities parseHWCapabilities(ByteArrayView capabilities);
Capabilities parseSWCapabilities(ByteArrayView capabilities);

}  // namespace tlv

}  // namespace compat
