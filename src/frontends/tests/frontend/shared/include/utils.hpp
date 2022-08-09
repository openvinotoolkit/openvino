// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <openvino/frontend/manager.hpp>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "ngraph/util.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/file_util.hpp"

// Helper functions
namespace FrontEndTestUtils {
std::string find_ov_path();
int run_tests(int argc, char** argv);

std::string get_current_executable_path();

inline std::tuple<ov::frontend::FrontEnd::Ptr, ov::frontend::InputModel::Ptr>
load_from_file(ov::frontend::FrontEndManager& fem, const std::string& frontend_name, const std::string& model_file) {
    auto frontend = fem.load_by_framework(frontend_name);
    auto inputModel = frontend->load(model_file);
    return std::tuple<ov::frontend::FrontEnd::Ptr, ov::frontend::InputModel::Ptr>{frontend, inputModel};
}

inline std::string fileToTestName(const std::string& fileName) {
    // TODO: GCC 4.8 has limited support of regex
    // return std::regex_replace(fileName, std::regex("[/\\.]"), "_");
    std::string res = fileName;
    for (auto& c : res) {
        if (c == '/') {
            c = '_';
        } else if (c == '.') {
            c = '_';
        }
    }
    return res;
}

inline int set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    return _putenv_s(name, value);
#elif defined(__linux) || defined(__APPLE__)
    std::string var = std::string(name) + "=" + value;
    return setenv(name, value, 0);
#endif
}

inline void setupTestEnv() {
    NGRAPH_SUPPRESS_DEPRECATED_START
    std::string fePath = ov::util::get_directory(find_ov_path());
    set_test_env("OV_FRONTEND_PATH", fePath.c_str());
    NGRAPH_SUPPRESS_DEPRECATED_END
}

inline bool exists(const std::string& file) {
    std::ifstream str(file, std::ios::in | std::ifstream::binary);
    return str.is_open();
}

inline std::string make_model_path(const std::string& modelsRelativePath) {
    return CommonTestUtils::getModelFromTestModelZoo(modelsRelativePath);
}

std::string get_disabled_tests(const std::string& manifest_path);
}  // namespace FrontEndTestUtils
