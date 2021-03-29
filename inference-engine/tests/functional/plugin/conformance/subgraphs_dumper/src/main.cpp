// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <regex>

#include "inference_engine.hpp"

#include "common_test_utils/file_utils.hpp"

#include "ops_cache.hpp"
#include "gflag_config.hpp"

// TODO: Poor exceptions handling
int main(int argc, char *argv[]) {
    uint8_t ret_code = 0;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return 0;
    }
    std::vector<std::string> input_folder_content;
    std::vector<std::string> dirs = CommonTestUtils::splitStringByDelimiter(FLAGS_input_folders);
    for (const auto &dir : dirs) {
        if (!CommonTestUtils::directoryExists(dir)) {
            std::string msg = "Input directory (" + dir + ") doesn't not exist!";
            throw std::runtime_error(msg);
        }
        CommonTestUtils::directoryFileListRecursive(dir, input_folder_content);
    }

    if (!CommonTestUtils::directoryExists(FLAGS_output_folder)) {
        std::string msg = "Output directory (" + FLAGS_output_folder + ") doesn't not exist!";
        throw std::runtime_error(msg);
    }

    auto ie = InferenceEngine::Core();
    auto cache = SubgraphsDumper::OPCache::make_cache();

    auto xml_regex = std::regex(R"(.*\.xml)");
    for (const auto &file : input_folder_content) {
        try {
            if (CommonTestUtils::fileExists(file) && std::regex_match(file, xml_regex)) {
                std::cout << "Processing model: " << file << std::endl;
                std::string bin_file = CommonTestUtils::replaceExt(file, "bin");
                if (!CommonTestUtils::fileExists(bin_file)) {
                    std::cerr << "Corresponding .bin file for the model " << file << " doesn't exist" << std::endl;
                    continue;
                }
                InferenceEngine::CNNNetwork net = ie.ReadNetwork(file);
                auto function = net.getFunction();
                cache->update_ops_cache(function, file);
            }
        } catch (std::exception &e) {
            std::cerr << "Model processing failed with exception:" << std::endl << e.what() << std::endl;
            ret_code = 1;
            continue;
        }
    }

    cache->serialize_cached_ops(FLAGS_output_folder);

    return ret_code;
}
