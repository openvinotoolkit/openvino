// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <manager.hpp>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "ngraph/util.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/file_util.hpp"
#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#        error "Only WINAPI_PARTITION_DESKTOP is supported, because of LoadLibrary[A|W]"
#    endif
#elif defined(__linux) || defined(__APPLE__)
#    include <dlfcn.h>
#endif

namespace {
inline std::string find_my_pathname() {
#ifdef _WIN32
    HMODULE hModule = GetModuleHandleW(SHARED_LIB_PREFIX L"ov_runtime" SHARED_LIB_SUFFIX);
    WCHAR wpath[MAX_PATH];
    GetModuleFileNameW(hModule, wpath, MAX_PATH);
    wstring ws(wpath);
    string path(ws.begin(), ws.end());
    replace(path.begin(), path.end(), '\\', '/');
    NGRAPH_SUPPRESS_DEPRECATED_START
    path = file_util::get_directory(path);
    NGRAPH_SUPPRESS_DEPRECATED_END
    path += "/";
    return path;
#elif defined(__linux) || defined(__APPLE__)
    Dl_info dl_info;
    dladdr(reinterpret_cast<void*>(ngraph::to_lower), &dl_info);
    return ov::util::get_absolute_file_path(dl_info.dli_fname);
#else
#    error "Unsupported OS"
#endif
}
}  // namespace

// Helper functions
namespace FrontEndTestUtils {
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
    std::string fePath = ov::util::get_directory(find_my_pathname());
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
}  // namespace FrontEndTestUtils
