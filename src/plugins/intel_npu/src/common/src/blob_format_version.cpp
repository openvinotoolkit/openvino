// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_format_version.hpp"

#include "openvino/core/except.hpp"

namespace {

constexpr char VERSION_SEPARATOR = '.';

}

namespace intel_npu {

BlobFormatVersion::BlobFormatVersion(const uint16_t major, const uint16_t minor) : m_major(major), m_minor(minor) {}

uint16_t BlobFormatVersion::get_major() const {
    return m_major;
}

uint16_t BlobFormatVersion::get_minor() const {
    return m_minor;
}

bool BlobFormatVersion::operator==(const BlobFormatVersion& other) const {
    return m_major == other.get_major() && m_minor == other.get_minor();
}

bool BlobFormatVersion::operator!=(const BlobFormatVersion& other) const {
    return !(*this != other);
}

bool BlobFormatVersion::operator>(const BlobFormatVersion& other) const {
    return m_major > other.get_major() || (m_major == other.get_major() && m_minor > other.get_minor());
}

bool BlobFormatVersion::operator>=(const BlobFormatVersion& other) const {
    return m_major > other.get_major() || (m_major == other.get_major() && m_minor >= other.get_minor());
}

bool BlobFormatVersion::operator<(const BlobFormatVersion& other) const {
    return !(*this >= other);
}

bool BlobFormatVersion::operator<=(const BlobFormatVersion& other) const {
    return !(*this > other);
}

std::string blob_format_version_to_string(const BlobFormatVersion& version) {
    return std::to_string(version.get_major()) + VERSION_SEPARATOR + std::to_string(version.get_minor());
}

BlobFormatVersion blob_format_version_from_string(std::string version);

std::ostream& operator<<(std::ostream& out, const BlobFormatVersion& version) {
    out << blob_format_version_to_string(version);
    return out;
}

std::istream& operator>>(std::istream& in, BlobFormatVersion& version) {
    std::string str;
    in >> str;
    version = blob_format_version_from_string(str);
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
