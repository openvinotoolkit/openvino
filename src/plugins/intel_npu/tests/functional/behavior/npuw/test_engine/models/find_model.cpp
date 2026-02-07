// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "openvino/util/file_util.hpp"

#include "find_model.hpp"

const std::string find_model(const std::string& model_name) {
    const char* pdir = std::getenv("NPU_TESTS_MODELS_PATH");
    if (pdir != nullptr) {
        std::filesystem::path dir(pdir);
        auto openvino_xml_full_path = dir / model_name / "openvino_model.xml";
        if (!ov::util::file_exists(openvino_xml_full_path)) {
            std::cout << "[INFO] Model is not found by \"" << openvino_xml_full_path
                      << "\" path" << std::endl;
             return "";
        }
        return openvino_xml_full_path.string();
    } else {
        std::cout << "[INFO] Environment variable \"NPU_TESTS_MODELS_PATH\" is not set, "
                  << "models' location is unknown!" << std::endl;
    }
    return "";
}
