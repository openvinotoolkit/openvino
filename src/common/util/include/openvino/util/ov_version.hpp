// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "openvino/core/except.hpp"

namespace ov::util {
/**
 * @brief Represents OpenVINO version with major, minor, patch, tweak, and build numbers.
 *
 * This structure parses and stores version information following the pattern:
 * MAJOR.MINOR.PATCH[.TWEAK]-BUILD-...
 * or a standalone build number.
 *
 * The version components can be compared using standard comparison operators.
 */
struct Version {
    size_t major = 0;
    size_t minor = 0;
    size_t patch = 0;
    size_t tweak = 0;
    size_t build = 0;

    explicit Version(const char* version_str) {
        // Pattern: MAJOR.MINOR.PATCH[.TWEAK]-BUILD-...
        std::regex full_pattern(R"(^([0-9]+)\.([0-9]+)\.([0-9]+)(?:\.([0-9]+))?\-([0-9]+)\-.*)");
        std::regex build_only_pattern(R"(^[0-9]+$)");
        std::cmatch match;

        if (std::regex_match(version_str, match, full_pattern)) {
            major = std::stoi(match[1].str());
            minor = std::stoi(match[2].str());
            patch = std::stoi(match[3].str());

            // match[4] is the entire optional group (\.TWEAK), match[5] is TWEAK
            // the 4th version number is not used in OpenVINO, but used in OpenVINO extensions
            if (match[4].matched) {
                tweak = std::stoi(match[4].str());
            }

            build = std::stoi(match[5].str());
        } else if (std::regex_match(version_str, build_only_pattern)) {
            // Parse as just a build number
            build = std::stoi(version_str);
        } else {
            OPENVINO_THROW("Failed to parse version: ", version_str);
        }
    }

    explicit Version(std::string_view version_str) : Version(version_str.data()) {}

    // Comparison operators
    bool operator==(const Version& other) const {
        return std::tie(major, minor, patch, tweak, build) ==
               std::tie(other.major, other.minor, other.patch, other.tweak, other.build);
    }

    bool operator!=(const Version& other) const {
        return !(*this == other);
    }

    bool operator<(const Version& other) const {
        return std::tie(major, minor, patch, tweak, build) <
               std::tie(other.major, other.minor, other.patch, other.tweak, other.build);
    }

    bool operator>(const Version& other) const {
        return other < *this;
    }

    bool operator<=(const Version& other) const {
        return !(other < *this);
    }

    bool operator>=(const Version& other) const {
        return !(*this < other);
    }
};

/**
 * @brief Policy for determining version compatibility between two versions.
 *
 * This structure defines the maximum allowed difference for each version component.
 * By default, all differences are allowed.
 */
struct VersionCompatibilityPolicy {
    size_t max_major_diff = std::numeric_limits<size_t>::max();
    size_t max_minor_diff = std::numeric_limits<size_t>::max();
    size_t max_patch_diff = std::numeric_limits<size_t>::max();
    size_t max_tweak_diff = std::numeric_limits<size_t>::max();
    size_t max_build_diff = std::numeric_limits<size_t>::max();
};

/**
 * @brief Checks if two versions are compatible based on the given policy.
 *
 * This function determines whether an older version is compatible with a newer version
 * by checking if the difference in each version component (major, minor, patch, tweak, build)
 * is within the limits specified by the compatibility policy.
 *
 * @param older_version The older version to check compatibility for.
 * @param newer_version The newer version to compare against.
 * @param policy The compatibility policy defining maximum allowed differences for each version component.
 *               Default policy allows all differences.
 * @return true if the versions are compatible according to the policy, false otherwise.
 *         Returns false if older_version is actually greater than newer_version.
 */
inline bool is_version_compatible(const Version& older_version,
                                  const Version& newer_version,
                                  const VersionCompatibilityPolicy& policy = {}) {
    if (older_version > newer_version) {
        return false;
    }

    // Check each version component against policy
    if (newer_version.major - older_version.major > policy.max_major_diff) {
        return false;
    }

    if (newer_version.minor - older_version.minor > policy.max_minor_diff) {
        return false;
    }

    if (newer_version.patch - older_version.patch > policy.max_patch_diff) {
        return false;
    }

    if (newer_version.tweak - older_version.tweak > policy.max_tweak_diff) {
        return false;
    }

    if (newer_version.build - older_version.build > policy.max_build_diff) {
        return false;
    }

    return true;
}
}  // namespace ov::util
