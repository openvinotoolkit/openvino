// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <regex>
#include <chrono>
#include <ctime>

#include "inference_engine.hpp"

#include "common_test_utils/file_utils.hpp"

#include "ops_cache.hpp"
#include "op_cloner.hpp"
#include "utils/dynamism_resolver.hpp"
#include "utils/model_wrap_struct.hpp"
#include "gflag_config.hpp"
#include <stdlib.h>
#include <string.h>

std::vector<SubgraphsDumper::Model> findModelsInDirs(const std::vector<std::string> &dirs) {
    std::vector<std::string> input_folder_content;
    for (const auto &dir : dirs) {
        if (!CommonTestUtils::directoryExists(dir)) {
            std::string msg = "Input directory (" + dir + ") doesn't not exist!";
            throw std::runtime_error(msg);
        }
        const auto content = CommonTestUtils::getFileListByPatternRecursive(dirs, std::regex(R"(.*\.xml)"));
        input_folder_content.insert(input_folder_content.end(), content.begin(), content.end());
    }
    std::vector<SubgraphsDumper::Model> models;
    auto xml_regex = std::regex(FLAGS_path_regex);
    for (const auto &file : input_folder_content) {
        if (std::regex_match(file, xml_regex)) {
            models.emplace_back(SubgraphsDumper::Model(file));
        }
    }
    std::sort(models.begin(), models.end());
    std::reverse(models.begin(), models.end());
    if (!CommonTestUtils::directoryExists(FLAGS_output_folder)) {
        std::string msg = "Output directory (" + FLAGS_output_folder + ") doesn't not exist!";
        throw std::runtime_error(msg);
    }
    return models;
}

void cacheModels(std::unique_ptr<SubgraphsDumper::OPCache> &cache,
                 uint8_t& ret_code,
                 const std::vector<SubgraphsDumper::Model>& models) {
    auto ie = InferenceEngine::Core();
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[20];
    size_t all_models = models.size();
    for (size_t i = 0; i < all_models; ++i) {
        const auto model = models[i];
        if (CommonTestUtils::fileExists(model.xml)) {
            try {
                time(&rawtime);
                timeinfo = localtime(&rawtime);  // NOLINT no localtime_r in C++11

                strftime(buffer, 20, "%H:%M:%S", timeinfo);
                std::cout << "[" << std::string(buffer) << "][" << i + 1 << "/" << all_models << "]Processing model: "
                          << model.xml << std::endl;
                if (!CommonTestUtils::fileExists(model.bin)) {
                    std::cout << "Corresponding .bin file for the model " << model.bin << " doesn't exist" << std::endl;
                    continue;
                }

                InferenceEngine::CNNNetwork net = ie.ReadNetwork(model.xml, model.bin);
                auto function = net.getFunction();
                if (FLAGS_eliminate_dynamism) {
                    try {
                        SubgraphsDumper::resolve_dynamic_shapes(function);
                    } catch (std::exception &e) {
                        std::cout << "Failed to eliminate dynamism from model " << model.xml
                                  << "\n Exception occurred:\n" << e.what() << "\nModel will be processed as is."
                                  << std::endl;
                    }
                }
                cache->update_ops_cache(function, model.xml);
            } catch (std::exception &e) {
                std::cout << "Model processing failed with exception:" << std::endl << e.what() << std::endl;
                ret_code = 1;
                continue;
            }
        }
    }
}


int main(int argc, char *argv[]) {
    uint8_t ret_code = 0;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return 0;
    }
    SubgraphsDumper::ClonersMap::constant_size_threshold_mb = FLAGS_constants_size_threshold;

    std::vector<std::string> local_cache_dirs = CommonTestUtils::splitStringByDelimiter(FLAGS_local_cache);
    std::vector<std::string> dirs = CommonTestUtils::splitStringByDelimiter(FLAGS_input_folders);
    auto cachedOps = findModelsInDirs(local_cache_dirs);
    auto models = findModelsInDirs(dirs);

    auto cache = SubgraphsDumper::OPCache::make_cache();
    cacheModels(cache, ret_code, cachedOps);
    cacheModels(cache, ret_code, models);
    cache->serialize_cached_ops(FLAGS_output_folder);

    return ret_code;
}