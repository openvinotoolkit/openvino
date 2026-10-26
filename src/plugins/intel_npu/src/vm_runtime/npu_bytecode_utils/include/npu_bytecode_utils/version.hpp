//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/span.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

namespace intel_npu::vm {

constexpr uint8_t VERSION_SIZE = 3;

class Version {
private:
    std::array<uint16_t, VERSION_SIZE> _version{0, 0, 0};

public:
    Version() = default;
    Version(uint16_t major, uint16_t minor, uint16_t patch): _version{major, minor, patch} {
    }

    static Version getMinSupportedVersion() {
        return Version(1, 0, 0);
    }

    uint16_t getMajor() const {
        return _version.at(0);
    }
    uint16_t getMinor() const {
        return _version.at(1);
    }
    uint16_t getPatch() const {
        return _version.at(2);
    }

    const std::array<uint16_t, VERSION_SIZE>& getFullVersion() const {
        return _version;
    }

    Version resetPatchNumber() const {
        return Version(getMajor(), getMinor(), 0);
    }

    bool operator<(const Version& other) const {
        return _version < other._version;
    };
    bool operator>(const Version& other) const {
        return _version > other._version;
    };
    bool operator<=(const Version& other) const {
        return _version <= other._version;
    };
    bool operator>=(const Version& other) const {
        return _version >= other._version;
    };
    bool operator==(const Version& other) const {
        return _version == other._version;
    };
    bool operator!=(const Version& other) const {
        return _version != other._version;
    };

    // Returns the total number of bytes used to represent the version
    static size_t getBinarySize();

    // Appends the version bytes to the provided buffer
    void appendTo(std::vector<uint8_t>& buffer) const;

    // Parses version bytes from the given buffer and populates the version fields.
    // Returns false and logs an error on malformed input. The buffer is advanced past the consumed bytes on success.
    bool parseFrom(vm::Span<uint8_t>& buffer);

    // Prints the version in "major.minor.patch" format to stdout
    void print(size_t indentLevel = 0) const;

    friend std::ostream& operator<<(std::ostream& os, const Version& v) {
        os << v.getMajor() << "." << v.getMinor() << "." << v.getPatch();
        return os;
    }
};

}  // namespace intel_npu::vm
