// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Core-local device identifiers. Minimal replacement for
// openvino/runtime/properties.hpp ov::device::UUID / ov::device::LUID.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ov::device {

struct UUID {
    static constexpr size_t MAX_UUID_SIZE = 16;
    std::array<uint8_t, MAX_UUID_SIZE> uuid = {};

    bool operator==(const UUID& other) const { return uuid == other.uuid; }
    bool operator!=(const UUID& other) const { return uuid != other.uuid; }
    bool operator<(const UUID& other) const { return uuid < other.uuid; }
};

struct LUID {
    static constexpr size_t MAX_LUID_SIZE = 8;
    std::array<uint8_t, MAX_LUID_SIZE> luid = {};

    bool operator==(const LUID& other) const { return luid == other.luid; }
    bool operator!=(const LUID& other) const { return luid != other.luid; }
    bool operator<(const LUID& other) const { return luid < other.luid; }
};

}  // namespace ov::device
