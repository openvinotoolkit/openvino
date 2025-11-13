//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <functional>

#include "concepts.hpp"
#include "check.hpp"
#include "registry.hpp"
#include "tlv.hpp"

#include <stdio.h>

namespace compat {

ByteArrayView::ByteArrayView(const uint8_t* ptr, size_t size) : ptr(ptr), size(size) {
    assert(ptr != nullptr && size > 0);
}

namespace {

bool hasUnknownUsage(const Capabilities& capabilities, const Requirements& requirements) {
    if (requirements.size() > capabilities.size()) {
        return true;
    }

    const auto isUnknown = [&capabilities](const auto& usage) {
        return capabilities.count(std::get<0>(usage)) == 0;
    };

    return std::any_of(std::begin(requirements), std::end(requirements), isUnknown);
}

}  // namespace

bool isCompatible(const Capabilities& capabilities, const Requirements& requirements) {
    printf("!!! Running isCompatible... !!!\n");
    if (hasUnknownUsage(capabilities, requirements)) {
        // it could be unsupported forward-compatibility case
        return false;
    }
    printf("!!!   No unknown usages !!!\n");

    // if blob doesn't say anything about something that must be validated - it's incompatible
    // so you can't just go over blob entries

    const auto isUnused = [&requirements](auto capability) {
        return requirements.count(capability) == 0;
    };

    const auto& registry = Registry::get();

    for (const auto& entry : capabilities) {
        printf("!!!   Processing a capability !!!\n");
        const auto header = std::get<0>(entry);
        const auto capability = std::get<1>(entry).get();

        // const auto stateless = header.length == 0;
        const auto required = header.required;
        // const auto optional = !required;
        const auto unused = isUnused(header);
        const auto registered = registry.isRegistered(header.type);

        // if capability is unregistered we cannot trigger validation
        // the only thing known about capability is required or not
        // allow only unused and unknown capabilities as in this case
        // result is pre-determined:
        // required & unused -> incompatible; optional & unused -> compatible
        assert(registered || unused);

        printf("!!!     It's registered !!!\n");

        if (required && unused) {
            printf("!!!     It's required & unused -> incompatible !!!\n");
            return false;
        }

        if (unused) {
            // optional capability is allowed to be unused
            printf("!!!     It's optional & unused -> ignoring !!!\n");
            continue;
        }

        const auto usage = requirements.at(header);
        assert(usage != nullptr);
        if (!capability->isSatisfied(usage)) {
            printf("!!!     Check failed -> incompatible !!!\n");
            return false;
        }
        printf("!!!     Check passed -> compatible so far !!!\n");
    }

    // here compare multiple capabilities at once

    return true;
}

namespace tlv {

// pointer arithmetic is intentional and is by TLV design
// reinterpret_cast is safe and intentional as tlv::Header and
// Requirement classes binary representation is kept cross-compatible
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-pro-type-reinterpret-cast)

// TODO: check parsing for duplicated entries - throw?

namespace {

Capabilities parseCapabilities(ByteArrayView capabilities, bool isHW) {
    Capabilities systemInfo;

    if (capabilities.ptr == nullptr && capabilities.size == 0) {
        return systemInfo;
    }

    assert(capabilities.ptr != nullptr && capabilities.size > 0);
    assert(reinterpret_cast<uintptr_t>(capabilities.ptr) % alignof(Header) == 0);
    assert(capabilities.size > sizeof(Header));

    printf("!!! capabilities.ptr = %p !!!\n", capabilities.ptr);
    printf("!!! capabilities.size = %lu !!!\n", capabilities.size);

    const auto& registry = Registry::get();

    const auto end = capabilities.ptr + capabilities.size;
    auto position = capabilities.ptr;
    while (position < end) {
        position += reinterpret_cast<uintptr_t>(position) % alignof(Header);
        const auto header = *reinterpret_cast<const Header*>(position);
        position += sizeof(Header) + header.offset;

        const auto stateless = header.length == 0;

        printf("!!! position = %p !!!\n", position);
        printf("!!! header->length = %u !!!\n", header.length);
        printf("!!! end = %p !!!\n", end);
        assert(position + header.length <= end);

        const auto parser = isHW ? &Registry::parseHW : &Registry::parseSW;
        systemInfo.emplace(header, std::invoke(parser, registry, header.type, stateless ? nullptr : position));
        //     registry.parse(header->type, stateless ? nullptr : position));
        //     ? registry.parseHW(header->type, stateless ? nullptr : position)
        //     : registry.parseSW(header->type, stateless ? nullptr : position);
        // // systemInfo.emplace(*header, stateless ? nullptr : registry.parse(header->type, position));
        // systemInfo.emplace(*header, registry.parse(header->type, stateless ? nullptr : position));
        position += header.length;
    }

    return systemInfo;
}

}  // namespace

Requirements parseRequirements(ByteArrayView section) {
    Requirements blobInfo;

    if (section.ptr == nullptr && section.size == 0) {
        return blobInfo;
    }

    printf("!!! section.ptr = %p !!!\n", section.ptr);
    printf("!!! section.size = %lu !!!\n", section.size);

    assert(section.ptr != nullptr && section.size != 0);
    assert(reinterpret_cast<uintptr_t>(section.ptr) % alignof(Header) == 0);
    assert(section.size > sizeof(Header));

    const auto end = section.ptr + section.size;
    auto position = section.ptr;
    while (position < end) {
        position += reinterpret_cast<uintptr_t>(position) % alignof(Header);
        const auto header = *reinterpret_cast<const Header*>(position);
        position += sizeof(Header) + header.offset;

        const auto stateless = header.length == 0;

        printf("!!! position = %p !!!\n", position);
        printf("!!! header.offset = %u !!!\n", header.offset);
        printf("!!! header.length = %u !!!\n", header.length);
        printf("!!! end = %p !!!\n", end);

        assert(position + header.length <= end);

        blobInfo.emplace(header, stateless ? nullptr : reinterpret_cast<const concepts::Requirement*>(position));
        position += header.length;
    }

    return blobInfo;
}

Capabilities parseHWCapabilities(ByteArrayView capabilities) {
    return parseCapabilities(capabilities, true);
}

Capabilities parseSWCapabilities(ByteArrayView capabilities) {
    return parseCapabilities(capabilities, false);
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-pro-type-reinterpret-cast)

}  // namespace tlv
}  // namespace compat
