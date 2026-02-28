// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <iomanip>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"
// clang-format on

/**
 * @brief Print OV Parameters
 * @param reference on OV Parameter
 * @return void
 */

int main(int argc, char* argv[]) {
    try {
        std::cout << "OpenVINO Runtime: " << ov::get_openvino_version() << std::endl;
        ov::Core core;
        std::string models_path = "C:\\Users\\openvino-228v001\\wzx\\OV_FP16-4BIT_DEFAULT\\OV_FP16-4BIT_DEFAULT\\openvino_model.xml";

        ov::AnyMap config;
        config[ov::enable_weightless.name()] = true;
        config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SIZE;
        config[ov::cache_dir.name()] = "C:/Users/openvino-228v001/wzx/openvino/temp/ov_cache";
        config[ov::cache_model_path.name()] = models_path;
        if (models_path.size() >= 4 && models_path.substr(models_path.size() - 4) == ".xml") {
            config[ov::weights_path.name()] = models_path.substr(0, models_path.size() - 4) + ".bin";
        }

        auto model = core.read_model(models_path);

        // ⭐ 关键：只到 compile
        auto compiled_model = core.compile_model(model, "GPU", config);

        // 程序直接 sleep，不 infer
        std::this_thread::sleep_for(std::chrono::seconds(10));
        // std::string models_path = argv[1];
    } catch (const std::exception& ex) {
        std::cerr << std::endl << "Exception occurred: " << ex.what() << std::endl << std::flush;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
