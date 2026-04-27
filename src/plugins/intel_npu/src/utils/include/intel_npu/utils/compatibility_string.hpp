// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iomanip>
#include <sstream>

namespace {

constexpr size_t HEX_BASE = 16;
constexpr size_t HEX_BYTE_LENGTH = 2;

}  // namespace

namespace intel_npu {

// TODO place these in source files?
static inline std::string encode_compatibility_string(const std::string& decoded_string) {
    std::ostringstream encoded_stringstream;
    for (const auto unit : decoded_string) {
        // setw + setfill will make sure the values within the range 0-F are padded (e.g. a->0a)
        encoded_stringstream << std::hex << std::setw(HEX_BYTE_LENGTH) << std::setfill('0') << int(unit);
    }
    return encoded_stringstream.str();
}

static inline std::string decode_compatibility_string(const std::string& encoded_string) {
    std::string decoded_string;
    for (size_t unit_index = 0; unit_index < encoded_string.length(); unit_index += HEX_BYTE_LENGTH) {
        decoded_string += std::stoi(encoded_string.substr(unit_index, HEX_BYTE_LENGTH), nullptr, HEX_BASE);
    }
    return decoded_string;
}

}  // namespace intel_npu
