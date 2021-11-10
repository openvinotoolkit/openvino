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

#define MAX_FILE_NAME_SIZE 255
#define SHORT_HASH_SIZE 7

namespace LayerTestsUtils {

class ExternalNetworkTool;
enum class ExternalNetworkMode;

using ENT = ExternalNetworkTool;
using ENTMode = ExternalNetworkMode;

enum class ExternalNetworkMode {
    DISABLED,
    LOAD,
    DUMP,
    DUMP_MODELS_ONLY,
    DUMP_INPUTS_ONLY
};

class ExternalNetworkTool {
private:
    static ExternalNetworkMode mode;

    static std::string& modelsPath();

    static constexpr const char* modelsNamePrefix = "TestModel_";

    template <typename T>
    static std::vector<std::shared_ptr<ov::Node>> topological_name_sort(T root_nodes);

    static void writeToHashMap(const std::string &network_name, const std::string &hash);

    static std::string replaceInName(const std::string &network_name, const std::map<std::string, std::string> replace_map);

    static std::string eraseInName(const std::string &network_name, const std::vector<std::string> patterns);

    static std::string eraseRepeatedInName(const std::string &network_name, const std::vector<char> target_symbols);

    template<typename T = float>
    static void writeToFile(const std::string &fileName, const T *ptrMemory, uint32_t numRows, uint32_t numColumns, std::string extension) {
        if (extension == "ark") {
            writeToArkFile(fileName, ptrMemory, numRows, numColumns);
        } else {
            printf("%s extension not supported", extension.c_str());
            return;
        }
    }

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

    template<typename T = float>
    static void readFromArkFile(const std::string fileName,
                                uint32_t arrayIndex,
                                std::string& ptrName,
                                std::vector<uint8_t>& memory,
                                uint32_t* ptrNumRows,
                                uint32_t* ptrNumColumns,
                                uint32_t* ptrNumBytesPerElement) {
        std::ifstream in_file(fileName.c_str(), std::ios::binary);
        if (in_file.good()) {
            uint32_t i = 0;
            while (i < arrayIndex) {
                std::string line;
                uint32_t numRows = 0u, numCols = 0u;
                std::getline(in_file, line, '\0');  // read variable length name followed by space and NUL
                std::getline(in_file, line, '\4');  // read "BFM" followed by space and control-D
                if (line.compare("BFM ") != 0) {
                    break;
                }
                in_file.read(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));  // read number of rows
                std::getline(in_file, line, '\4');                                  // read control-D
                in_file.read(reinterpret_cast<char*>(&numCols), sizeof(uint32_t));  // read number of columns
                in_file.seekg(numRows * numCols * sizeof(float), in_file.cur);      // read data
                i++;
            }
            if (!in_file.eof()) {
                std::string line;
                std::getline(in_file, ptrName, '\0');  // read variable length name followed by space and NUL
                std::getline(in_file, line, '\4');     // read "BFM" followed by space and control-D
                if (line.compare("BFM ") != 0) {
                    throw std::runtime_error(std::string("Cannot find array specifier in file %s in LoadFile()!\n") +
                                            fileName);
                }
                in_file.read(reinterpret_cast<char*>(ptrNumRows), sizeof(uint32_t));     // read number of rows
                std::getline(in_file, line, '\4');                                       // read control-D
                in_file.read(reinterpret_cast<char*>(ptrNumColumns), sizeof(uint32_t));  // read number of columns
                in_file.read(reinterpret_cast<char*>(&memory.front()),
                            *ptrNumRows * *ptrNumColumns * sizeof(float));  // read array data
            }
            in_file.close();
        } else {
            throw std::runtime_error(std::string("Failed to open %s for reading in LoadFile()!\n") + fileName);
        }

        *ptrNumBytesPerElement = sizeof(float);
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

    static void unifyFunctionNames(std::shared_ptr<ngraph::Function> network);

    static void saveInputFile(const std::string &network_name,
                              const InferenceEngine::InputInfo::CPtr &input_info,
                              const InferenceEngine::Blob::Ptr &blob,
                              uint32_t id,
                              std::string extension = "ark");

    static const std::string &getModelsPath() { return modelsPath(); }

    static ExternalNetworkMode getMode() { return mode; }

    static bool isMode(ExternalNetworkMode val) { return mode == val; }

    static void setModelsPath(std::string &val);

    static void setMode(ExternalNetworkMode val) { mode = val; }

    static std::string processTestName(const std::string &network_name, const size_t extension_len = 3);

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