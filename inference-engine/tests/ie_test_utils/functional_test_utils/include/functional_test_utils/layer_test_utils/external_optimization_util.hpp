// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <map>
#include <fstream>
#include <unordered_map>
#include <common_test_utils/file_utils.hpp>
#include <ie_core.hpp>
#include <ie_common.h>
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/graph_util.hpp"
#include "transformations/serialize.hpp"
#include "cpp/ie_cnn_network.h"

namespace LayerTestsUtils {

#define SKIP_VALIDATION_IF_OPTIMIZATION_MODE_IS_DUMP()                  \
if (ExtOptUtil::toDumpModel() || ExtOptUtil::toDumpInput()) {           \
    return;                                                             \
}                                                                       \

class ExternalOptimizationUtil;
enum class ExternalOptimizationMode;

using ExtOptUtil = ExternalOptimizationUtil;
using ExtOptMode = ExternalOptimizationMode;

enum class ExternalOptimizationMode {
    DISABLED,
    LOAD,
    DUMP,
    DUMP_MODELS_ONLY,
    DUMP_INPUTS_ONLY,
    DUMP_ALL,
};

class ExternalOptimizationUtil {
#define MAX_FILE_NAME_SIZE 240
#define SHORT_HASH_SIZE 10
#define DEFAULT_INPUT_EXTENSION "bin"

#define path_delimiter "/"
#ifdef _WIN32
#define path_delimiter "\\"
#endif

private:
    static ExternalOptimizationMode mode;

    static std::string& modelsPath();

    static constexpr const char* modelsNamePrefix = "TestModel_";

    static constexpr const char* modelsHashPrefix = "hash";

    template <typename T>
    static std::vector<std::shared_ptr<ov::Node>> topological_name_sort(T root_nodes);

    static void writeToHashMap(const std::string &network_name, const std::string &hash);

    static std::string replaceInName(const std::string &network_name, const std::map<std::string, std::string> replace_map);

    static std::string eraseInName(const std::string &network_name, const std::vector<std::string> patterns);

    static std::string eraseRepeatedInName(const std::string &network_name, const std::vector<char> target_symbols);

    template<typename T>
    static void writeDataToFile(const std::string &fileName, const T *ptrMemory, uint32_t numRows, uint32_t numColumns, std::string extension) {
        try {
            if (extension == "ark") {
                writeToArkFile(fileName, ptrMemory, numRows, numColumns);
            } else if (extension == "bin") {
                writeToBinaryFile(fileName, ptrMemory, numRows * numColumns);
            } else {
                printf("%s extension not supported", extension.c_str());
                return;
            }
        } catch (const std::exception &e) {
            printf("Failed to write data to file with exception: %s\n", e.what());
        }
    }

    template<typename T>
    static void writeToArkFile(const std::string &fileName, const T *ptrMemory, uint32_t numRows, uint32_t numColumns) {
        throw std::runtime_error(std::string("Ark files support only floating point data\n") + fileName);
    }

    template<typename T>
    static void writeToBinaryFile(const std::string &fileName, const T *ptrMemory, uint32_t elements_number) {
        std::ios_base::openmode mode = std::ios::binary;
        std::ofstream out_file(fileName.c_str(), mode);
        if (out_file.good()) {
            out_file.write(reinterpret_cast<const char*>(ptrMemory), elements_number * sizeof(T));
            out_file.close();
        } else {
            throw std::runtime_error(std::string("Failed to open %s for writing in writeToBinaryFile()!\n") + fileName);
        }
        printf("Input data dumped to binary file %s\n", fileName.c_str());
    }


protected:
    ExternalOptimizationUtil() = delete;

    ~ExternalOptimizationUtil() = delete;

public:
    static void dumpNetworkToFile(const std::shared_ptr<ngraph::Function> network,
                           const std::string &network_name);

    static InferenceEngine::CNNNetwork loadNetworkFromFile(const std::shared_ptr<InferenceEngine::Core> core,
                                                    const std::string &network_name);

    static std::shared_ptr<ngraph::Function> loadNetworkFromFile(const std::string &network_name);

    static void updateFunctionNames(std::shared_ptr<ngraph::Function> network);

    static void unifyFunctionNames(std::shared_ptr<ngraph::Function> network);

    static void saveInputFile(const std::string &network_name,
                              const InferenceEngine::InputInfo::CPtr &input_info,
                              const InferenceEngine::Blob::CPtr &blob,
                              uint32_t id,
                              std::string extension = DEFAULT_INPUT_EXTENSION);

    static const std::string &getModelsPath() { return modelsPath(); }

    static ExternalOptimizationMode getMode() { return mode; }

    static bool isMode(ExternalOptimizationMode val) { return mode == val; }

    static bool toDumpModel();

    static bool toDumpInput();

    static bool toLoad();

    static void setModelsPath(std::string &val);

    static void setMode(ExternalOptimizationMode val) { mode = val; }

    static std::string processTestName(const std::string &network_name, const size_t extension_len = 3);

    static std::string generateHashName(std::string value);
};

}  // namespace LayerTestsUtils