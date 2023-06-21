// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"
#include "openvino/util/file_util.hpp"

ov::Exception::Exception(const std::string& what_arg) : std::runtime_error(what_arg) {}

void ov::Exception::create(const CheckLocInfo& check_loc_info, const std::string& explanation) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    throw ov::Exception(make_what(check_loc_info, default_msg, explanation));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

std::string ov::Exception::make_what(const CheckLocInfo& check_loc_info,
                                     const std::string& context_info,
                                     const std::string& explanation) {
    // Use relative path only for internal code
    auto getRelativePath = [](const std::string& path) -> std::string {
        // Path to local OpenVINO repository
        static const std::string project_root(PROJECT_ROOT_DIR);
        auto root_path = util::sanitize_path(project_root);
#ifdef _WIN32
        util::convert_path_win_style(root_path);
#endif
        auto p = util::sanitize_path(path);

        std::cout << util::sanitize_path(PROJECT_ROOT_DIR) << std::endl;
        util::convert_path_win_style(root_path);

        // All internal paths start from project root
        if (p.find(root_path) != 0) {
            std::cout << "ret 1\n";
            return path;
        }
        // Add +1 to remove first /
        return p.substr(root_path.length() + 1);
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

const std::string ov::Exception::default_msg{};

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

const std::string ov::NotImplemented::default_msg{"Not Implemented"};
