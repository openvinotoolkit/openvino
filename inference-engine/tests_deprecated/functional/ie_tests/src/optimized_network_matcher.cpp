// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <string>
#include <gtest/gtest.h>
#include <ie_plugin_config.hpp>
#include "optimized_network_matcher.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;


void Regression :: Matchers :: OptimizedNetworkMatcher :: to(std::string path_to_reference_dump) {
    ModelsPath path_to_firmware;
    path_to_firmware << kPathSeparator << config._firmware << kPathSeparator;

    auto compact_token = config.compactMode ? "_compact" : "";

    this->path_to_reference_dump = path_to_firmware + path_to_reference_dump + compact_token + "_firmware.bin";
}

void Regression :: Matchers :: OptimizedNetworkMatcher :: matchCustom ()  {
    ASSERT_NO_FATAL_FAILURE(createExecutableNetworkFromIR());
    firmware = readDumpFromFile(config._tmp_firmware);
    ASSERT_NE(firmware.size(), 0);
}

std::vector<uint8_t> Regression :: Matchers :: OptimizedNetworkMatcher :: readDumpFromFile(std::string path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    if (size <=0) {
        return std::vector<uint8_t>();
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    file.read((char*)buffer.data(), size);

    return buffer;
}

void Regression :: Matchers :: OptimizedNetworkMatcher :: checkResult() {
    auto refFirmware = readDumpFromFile(path_to_reference_dump);

    ASSERT_EQ(refFirmware.size(), firmware.size()) << "Reference: " << path_to_reference_dump;

    for (int i = 0; i < refFirmware.size(); ++i) {
        ASSERT_EQ(refFirmware[i], firmware[i]) << "firmware mismatch at: " << i << " byte";
    }
}

////////////////////////////

void Regression :: Matchers :: OptimizedNetworkDumper::dump()  {
    ExecutableNetwork executableApi;
    ASSERT_NO_FATAL_FAILURE(executableApi = createExecutableNetworkFromIR());
    try {
        executableApi.Export(config._path_to_aot_model);
    }
    catch (const std::exception &e) {
         FAIL() << e.what();
    }

}
