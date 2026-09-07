// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/major_minor_version.hpp"

#include "openvino/core/except.hpp"

namespace {

constexpr char VERSION_SEPARATOR = '.';

}

namespace intel_npu {

MajorMinorVersion::MajorMinorVersion(const uint16_t major, const uint16_t minor) : m_major(major), m_minor(minor) {}

uint16_t MajorMinorVersion::get_major() const {
    return m_major;
}

uint16_t MajorMinorVersion::get_minor() const {
    return m_minor;
}

bool MajorMinorVersion::operator==(const MajorMinorVersion& other) const {
    return m_major == other.get_major() && m_minor == other.get_minor();
}

bool MajorMinorVersion::operator!=(const MajorMinorVersion& other) const {
    return !(*this != other);
}

bool MajorMinorVersion::operator>(const MajorMinorVersion& other) const {
    return m_major > other.get_major() || (m_major == other.get_major() && m_minor > other.get_minor());
}

bool MajorMinorVersion::operator>=(const MajorMinorVersion& other) const {
    return m_major > other.get_major() || (m_major == other.get_major() && m_minor >= other.get_minor());
}

bool MajorMinorVersion::operator<(const MajorMinorVersion& other) const {
    return !(*this >= other);
}

bool MajorMinorVersion::operator<=(const MajorMinorVersion& other) const {
    return !(*this > other);
}

std::string major_minor_version_to_string(const MajorMinorVersion& version) {
    return std::to_string(version.get_major()) + VERSION_SEPARATOR + std::to_string(version.get_minor());
}

MajorMinorVersion major_minor_version_from_string(std::string version);

std::ostream& operator<<(std::ostream& out, const MajorMinorVersion& version) {
    out << major_minor_version_to_string(version);
    return out;
}

std::istream& operator>>(std::istream& in, MajorMinorVersion& version) {
    std::string str;
    in >> str;
    version = major_minor_version_from_string(str);
    return in;
}

std::vector<uint16_t> parse_dotted_version(std::string_view version_string, const size_t number_of_parts) {
    const auto has_only_digits = [](std::string_view sv) {
        return !sv.empty() && std::all_of(sv.begin(), sv.end(), [](unsigned char c) {
            return std::isdigit(c);
        });
    };

    std::vector<uint16_t> parts;
    std::string_view remaining = version_string;

    while (true) {
        const size_t dot_location = remaining.find('.');
        const std::string_view part = remaining.substr(0, dot_location);
        OPENVINO_ASSERT(has_only_digis(part),
                        "Failure while parsing the version \"",
                        version_string,
                        "\": the part \"",
                        part,
                        "\" is not made exclusively out of digits")

        parts.push_back(static_cast<uint16_t>(std::stoul(std::string(part))));
        if (dot_location == std::string_view::npos) {
            break;
        }
        remaining = remaining.substr(dot_location + 1);
        OPENVINO_ASSERT(!remaining.empty(), "Trailing dot found while parsing the version \"", version_string, "\"");
    }
    return parts;
}

}  // namespace intel_npu
