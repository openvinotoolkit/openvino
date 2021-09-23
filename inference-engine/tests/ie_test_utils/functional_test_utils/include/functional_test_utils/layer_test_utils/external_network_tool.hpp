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

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/graph_util.hpp"
#include "transformations/serialize.hpp"
#include "cpp/ie_cnn_network.h"
#include <ie_core.hpp>
#include <ie_common.h>

namespace LayerTestsUtils {

class ExternalNetworkTool;

class ExternalNetworkToolDestroyer {
private:
    ExternalNetworkTool *p_instance;
public:
    ~ExternalNetworkToolDestroyer();

    void initialize(ExternalNetworkTool *p);
};

enum class ExternalNetworkMode {
    DISABLED,
    IMPORT,
    EXPORT
};

class ExternalNetworkTool {
private:
    static ExternalNetworkTool *p_instance;
    static ExternalNetworkToolDestroyer destroyer;

    static ExternalNetworkMode mode;
    static const char *modelsPath;

    friend class ExternalNetworkToolDestroyer;

    template <typename T>
    static std::vector<std::shared_ptr<ov::Node>> topological_name_sort(T root_nodes);

    static void writeToHashMap(const std::string &network_name, const std::string &hash);

protected:
    ExternalNetworkTool() = default;

    ~ExternalNetworkTool() = default;

public:
    void dumpNetworkToFile(const std::shared_ptr<ngraph::Function> network,
                           const std::string &network_name) const;

    InferenceEngine::CNNNetwork loadNetworkFromFile(const std::shared_ptr<InferenceEngine::Core> core,
                                                    const std::string &network_name) const;

    std::shared_ptr<ngraph::Function> loadNetworkFromFile(const std::string &network_name) const;

    void updateFunctionNames(std::shared_ptr<ngraph::Function> network) const;
    // std::shared_ptr<ngraph::Function> renameFunction(std::shared_ptr<ngraph::Function>) const;

    static ExternalNetworkTool &getInstance();

    static std::string getModelsPath() { return std::string(modelsPath); }

    static ExternalNetworkMode getMode() { return mode; }

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