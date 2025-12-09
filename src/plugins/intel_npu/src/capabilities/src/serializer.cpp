//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <stdio.h>

#include <cassert>
#include <cstdint>
#include <cstring>

#include "serializer.hpp"
#include "tlv.hpp"

namespace compat::tlv {

size_t alignment() {
    return alignof(compat::tlv::Header);
}

namespace {

uint8_t* appendTo(uint8_t* dst, const uint8_t* src, size_t size) {
    std::memcpy(dst, src, size);
    return dst + size;
}

template <class T>
uint8_t* appendObjectTo(uint8_t* dst, const T& object) {
    const auto objectBegin = reinterpret_cast<const uint8_t*>(&object);
    constexpr auto objectSize = sizeof(object);
    appendTo(dst, objectBegin, objectSize);
    return dst + objectSize;
}

}  // namespace

void Serializer::appendImpl(uint16_t id, bool policy) {
    appendImpl(id, policy, nullptr, 1, 0);
}

void Serializer::appendImpl(uint16_t id, bool policy, const uint8_t* ptr, size_t alignment, size_t size) {
    assert(alignment > 0);
    assert((ptr == nullptr && alignment == 1 && size == 0) || (ptr != nullptr && size > 0));

    const auto oldSize = _buffer.size();
    auto newSize = oldSize;

    const auto headerPadding = newSize % alignof(Header);
    newSize += headerPadding + sizeof(Header);

    const auto objectPadding = newSize % alignment;
    assert(objectPadding <= sizeof(uint64_t) - 1);
    newSize += objectPadding + size;

    _buffer.resize(newSize);

    auto dst = &_buffer[oldSize + headerPadding];
    dst = appendObjectTo(dst, Header{id, policy, static_cast<uint8_t>(objectPadding), static_cast<uint16_t>(size)});

    if (ptr == nullptr) {
        return;
    }

    dst += objectPadding;
    appendTo(dst, ptr, size);
}

std::vector<uint8_t> Serializer::done() {
    return std::move(_buffer);
}

}  // namespace tlv
