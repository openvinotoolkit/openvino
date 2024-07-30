// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <openvino/frontend/manager.hpp>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/file_util.hpp"

// Helper functions
namespace FrontEndTestUtils {

int run_tests(int argc, char** argv, const std::string& manifest);

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

inline bool exists(const std::string& file) {
    std::ifstream str(file, std::ios::in | std::ifstream::binary);
    return str.is_open();
}

inline std::string make_model_path(const std::string& modelsRelativePath) {
    return ov::test::utils::getModelFromTestModelZoo(modelsRelativePath);
}

std::string get_disabled_tests(const std::string& manifest_path);
}  // namespace FrontEndTestUtils
