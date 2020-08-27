//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/check.hpp"

using namespace ngraph;

std::string CheckFailure::make_what(const CheckLocInfo& check_loc_info,
                                    const std::string& context_info,
                                    const std::string& explanation)
{
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
    ss << "Check '" << check_loc_info.check_string << "' failed at "
       << getRelativePath(check_loc_info.file) << ":" << check_loc_info.line;
    if (!context_info.empty())
    {
        ss << ":" << std::endl << context_info;
    }
    if (!explanation.empty())
    {
        ss << ":" << std::endl << explanation;
    }
    ss << std::endl;
    return ss.str();
}
