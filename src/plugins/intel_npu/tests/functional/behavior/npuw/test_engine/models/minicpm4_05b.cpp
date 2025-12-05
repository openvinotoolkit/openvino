// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "openvino/util/file_util.hpp"

#include "minicpm4_05b.hpp"

const std::string get_minicpm4_05b_path() {
    static constexpr const char* model_name = "MiniCPM4-0.5B_int4_sym_group128_dyn_stateful";
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
