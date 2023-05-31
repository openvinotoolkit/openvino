// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"

ov::Exception::Exception(const std::string& what_arg) : std::runtime_error(what_arg) {}

ov::Exception::Exception(const std::stringstream& what_arg) : std::runtime_error(what_arg.str()) {}

void ov::Exception::create(const CheckLocInfo& check_loc_info,
                           const std::string& context_info,
                           const std::string& explanation) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    CheckLocInfo loc_info;
    loc_info.file = check_loc_info.file;
    loc_info.line = check_loc_info.line;
    loc_info.check_string = nullptr;
    throw ov::Exception(make_what(loc_info, context_info, explanation));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

std::string ov::Exception::make_what(const CheckLocInfo& check_loc_info,
                                     const std::string& context_info,
                                     const std::string& explanation) {
    // Use relative path only for internal code
    auto getRelativePath = [](const std::string& path) -> std::string {
        // Path to local OpenVINO repository
        static const std::string project_root(PROJECT_ROOT_DIR);
        // All internal paths start from project root
        if (path.find(project_root) != 0)
            return path;
        // Add +1 to remove first /
        return path.substr(project_root.length() + 1);
    };
    std::stringstream ss;
    if (check_loc_info.check_string) {
        ss << "Check '" << check_loc_info.check_string << "' failed at " << getRelativePath(check_loc_info.file) << ":"
           << check_loc_info.line;
    } else {
        ss << "Exception from " << getRelativePath(check_loc_info.file) << ":" << check_loc_info.line;
    }
    if (!context_info.empty()) {
        ss << ":" << std::endl << context_info;
    }
    if (!explanation.empty()) {
        ss << ":" << std::endl << explanation;
    }
    ss << std::endl;
    return ss.str();
}

ov::Exception::~Exception() = default;

void ov::AssertFailure::create(const CheckLocInfo& check_loc_info,
                               const std::string& context_info,
                               const std::string& explanation) {
    throw ov::AssertFailure(make_what(check_loc_info, context_info, explanation));
}
ov::AssertFailure::~AssertFailure() = default;

void ov::NotImplemented::create(const CheckLocInfo& check_loc_info,
                                const std::string& context_info,
                                const std::string& explanation) {
    throw ov::NotImplemented(make_what(check_loc_info, context_info, explanation));
}
ov::NotImplemented::~NotImplemented() = default;
