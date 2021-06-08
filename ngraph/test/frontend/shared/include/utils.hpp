// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

// Helper functions
namespace FrontEndTestUtils
{
    inline std::string fileToTestName(const std::string& fileName)
    {
        // TODO: GCC 4.8 has limited support of regex
        // return std::regex_replace(fileName, std::regex("[/\\.]"), "_");
        std::string res = fileName;
        for (auto& c : res)
        {
            if (c == '/')
            {
                c = '_';
            }
            else if (c == '.')
            {
                c = '_';
            }
        }
        return res;
    }
} // namespace FrontEndTestUtils