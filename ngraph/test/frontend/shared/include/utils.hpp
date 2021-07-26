// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <string>
#include "backend.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/file_util.hpp"

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

    inline int set_test_env(const char* name, const char* value)
    {
#ifdef _WIN32
        return _putenv_s(name, value);
#elif defined(__linux) || defined(__APPLE__)
        std::string var = std::string(name) + "=" + value;
        return setenv(name, value, 0);
#endif
    }

    inline void setupTestEnv()
    {
        std::string fePath = ngraph::file_util::get_directory(
            ngraph::runtime::Backend::get_backend_shared_library_search_directory());
        set_test_env("OV_FRONTEND_PATH", fePath.c_str());
    }

    inline bool exists(const std::string& file)
    {
        std::ifstream str(file, std::ios::in | std::ifstream::binary);
        return str.is_open();
    }

    inline std::string make_model_path(const std::string& modelsRelativePath)
    {
        // First try build path
        auto res = std::string(TEST_MODEL_BUILD_DIR) + "/" + modelsRelativePath;
        if (exists(res))
        {
            return res;
        }
        else
        {
            // Install case: if model file does not exist, use base path from env variable
            return std::string(ngraph::getenv_string("FE_TEST_MODELS")) + "/" + modelsRelativePath;
        }
    }
} // namespace FrontEndTestUtils