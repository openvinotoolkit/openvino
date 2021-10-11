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
#include <common_test_utils/file_utils.hpp>
#include <ie_core.hpp>
#include <ie_common.h>
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/graph_util.hpp"
#include "transformations/serialize.hpp"
#include "cpp/ie_cnn_network.h"

#define path_delimiter "/"
#ifdef _WIN32
#define path_delimiter "\\"
#endif

namespace LayerTestsUtils {

class ExternalNetworkTool;
enum class ExternalNetworkMode;

using ENT = ExternalNetworkTool;
using ENTMode = ExternalNetworkMode;

enum class ExternalNetworkMode {
    DISABLED,
    IMPORT,
    EXPORT,
    EXPORT_MODELS_ONLY,
    EXPORT_ARKS_ONLY
};

class ExternalNetworkTool {
private:
    static ExternalNetworkMode mode;
    static const char *modelsPath;

    template <typename T>
    static std::vector<std::shared_ptr<ov::Node>> topological_name_sort(T root_nodes);

    static void writeToHashMap(const std::string &network_name, const std::string &hash);

    template<typename T = float>
    static void writeToArkFile(const std::string &fileName, const T *ptrMemory, uint32_t numRows, uint32_t numColumns) {
        std::ios_base::openmode mode = std::ios::binary;
        std::ofstream out_file(fileName.c_str(), mode);
        const std::string &token = "input ";
        if (out_file.good()) {
            out_file.write(token.c_str(), token.length());
            out_file.write("\0", 1);
            out_file.write("BFM ", 4);
            out_file.write("\4", 1);
            out_file.write(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));
            out_file.write("\4", 1);
            out_file.write(reinterpret_cast<char*>(&numColumns), sizeof(uint32_t));
            out_file.write(reinterpret_cast<const char*>(ptrMemory), numRows * numColumns * sizeof(T));
            out_file.close();
        } else {
            throw std::runtime_error(std::string("Failed to open %s for writing in saveArkFile()!\n") + fileName);
        }
        printf("Input data dumped to ark file %s\n", fileName.c_str());
    }

protected:
    ExternalNetworkTool() = delete;

    ~ExternalNetworkTool() = delete;

public:
    static void dumpNetworkToFile(const std::shared_ptr<ngraph::Function> network,
                           const std::string &network_name);

    static InferenceEngine::CNNNetwork loadNetworkFromFile(const std::shared_ptr<InferenceEngine::Core> core,
                                                    const std::string &network_name);

    static std::shared_ptr<ngraph::Function> loadNetworkFromFile(const std::string &network_name);

    static void updateFunctionNames(std::shared_ptr<ngraph::Function> network);

    static void saveArkFile(const std::string &network_name,
                            const InferenceEngine::InputInfo::CPtr &input_info,
                            const InferenceEngine::Blob::Ptr &blob,
                            uint32_t id);

    static std::string getModelsPath() { return std::string(modelsPath); }

    static ExternalNetworkMode getMode() { return mode; }

    static bool isMode(ExternalNetworkMode val) { return mode == val; }

    static void setModelsPath(std::string &val) {
        modelsPath = val.c_str();
    }

    static void setMode(ExternalNetworkMode val) { mode = val; }

    static std::string generateHashName(std::string value) {
        auto command = "python sha256hash.py " + value;
        auto hash = executeCommand(command);
        hash.resize(hash.length() - 1);
        return hash;
    }

    static std::string executeCommand(std::string cmd) {
        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        return result;
    }
};

}  // namespace LayerTestsUtils