// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace ov::util {
struct OVVersion {
    int major = 0;
    int minor = 0;
    int patch = 0;
    int tweak = 0;
    int build = 0;

    OVVersion(const std::string& version_str) {
        // Pattern: MAJOR.MINOR.PATCH[.TWEAK]-BUILD-...
        std::regex full_pattern(R"(^([0-9]+)\.([0-9]+)\.([0-9]+)(\.([0-9]+))?\-([0-9]+)\-.*)");
        std::regex build_only_pattern(R"(^[0-9]+$)");
        std::smatch match;
        
        if (std::regex_match(version_str, match, full_pattern)) {
            major = std::stoi(match[1].str());
            minor = std::stoi(match[2].str());
            patch = std::stoi(match[3].str());
            
            // match[4] is the entire optional group (\.TWEAK), match[5] is TWEAK
            // the 4th version number is not used in OpenVINO, but used in OpenVINO extensions
            if (match[5].matched) {
                tweak = std::stoi(match[5].str());
            }
            
            build = std::stoi(match[6].str());
        }
        else if (std::regex_match(version_str, build_only_pattern)) {
            // Parse as just a build number
            build = std::stoi(version_str);
        }
        else {
            OPENVINO_THROW("Failed to parse version: ", version_str);
        }
    }

    OVVersion(const char* version_str) : OVVersion(std::string(version_str)) {};

    // Comparison operators
    bool operator==(const OVVersion& other) const {
        return major == other.major && 
               minor == other.minor && 
               patch == other.patch && 
               tweak == other.tweak && 
               build == other.build;
    }

    bool operator!=(const OVVersion& other) const {
        return !(*this == other);
    }

    bool operator<(const OVVersion& other) const {
        if (major != other.major) return major < other.major;
        if (minor != other.minor) return minor < other.minor;
        if (patch != other.patch) return patch < other.patch;
        if (tweak != other.tweak) return tweak < other.tweak;
        return build < other.build;
    }

    bool operator>(const OVVersion& other) const {
        return other < *this;
    }

    bool operator<=(const OVVersion& other) const {
        return !(other < *this);
    }

    bool operator>=(const OVVersion& other) const {
        return !(*this < other);
    }
};

bool is_compiled_blob_compatible(const OVVersion& compiled_blob_version, const OVVersion& runtime_version) {
    if (compiled_blob_version > runtime_version) {
        return false;
    }

    // Major version must be equal
    if (compiled_blob_version.major != runtime_version.major) {
        return false;
    }
    
    // compiled blob minor version can be lower than runtime minor version up to 5
    if (runtime_version.minor - compiled_blob_version.minor >= 5) {
        return false;
    }

    return true;
}
} // namespace ov::util
