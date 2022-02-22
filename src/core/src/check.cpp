// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/check.hpp"

using namespace ngraph;

std::string CheckFailure::make_what(const CheckLocInfo& check_loc_info,
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
    ss << "Check '" << check_loc_info.check_string << "' failed at " << getRelativePath(check_loc_info.file) << ":"
       << check_loc_info.line;
    if (!context_info.empty()) {
        ss << ":" << std::endl << context_info;
    }
    if (!explanation.empty()) {
        ss << ":" << std::endl << explanation;
    }
    ss << std::endl;
    return ss.str();
}
