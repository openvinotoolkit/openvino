// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

namespace intel_npu {

/**
 * @brief Contains a major and a minor version; provides a set of operators for handling them.
 */
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

    std::string to_string() const;

    static MajorMinorVersion from_string(const std::string_view version);

private:
    uint16_t m_major;
    uint16_t m_minor;
};

const MajorMinorVersion CURRENT_BLOB_FORMAT_VERSION(3, 0);
const MajorMinorVersion CURRENT_RUNTIME_REQUIREMENTS_VERSION(3, 0);

std::ostream& operator<<(std::ostream& out, const MajorMinorVersion& version);

std::istream& operator>>(std::istream& is, MajorMinorVersion& version);

/**
 * @brief Parses a string of the form <uint>.<uint>.[...]. The number of parsed integers have to be equal to the
 * expected number given as argument.
 *
 * @throws ov::Exception if the format is invalid
 *
 * @param version_string The string to be parsed
 * @param number_of_parts The expected number of integers contained in the string
 * @return All integers found in the string
 */
std::vector<uint16_t> parse_dotted_version(std::string_view version_string, const size_t number_of_parts);

}  // namespace intel_npu
