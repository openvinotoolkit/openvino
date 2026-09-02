// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

namespace intel_npu {

const MajorMinorVersion CURRENT_BLOB_FORMAT_VERSION(3, 0);
const MajorMinorVersion CURRENT_RUNTIME_REQUIREMENTS_VERSION(3, 0);

class MajorMinorVersion {
public:
    MajorMinorVersion(const uint16_t major, const uint16_t minor);

    uint16_t get_major() const;

    uint16_t get_minor() const;

    bool operator==(const MajorMinorVersion& other) const;

    bool operator!=(const MajorMinorVersion& other) const;

    bool operator>(const MajorMinorVersion& other) const;

    bool operator>=(const MajorMinorVersion& other) const;

    bool operator<(const MajorMinorVersion& other) const;

    bool operator<=(const MajorMinorVersion& other) const;

private:
    uint16_t m_major;
    uint16_t m_minor;
};

// TODO test these
std::string blob_format_version_to_string(const MajorMinorVersion& version);

MajorMinorVersion blob_format_version_from_string(std::string version);

std::ostream& operator<<(std::ostream& out, const MajorMinorVersion& version);

std::istream& operator>>(std::istream& is, MajorMinorVersion& version);

std::vector<uint16_t> parse_dotted_version(std::string_view version_string, const size_t number_of_parts);

}  // namespace intel_npu
