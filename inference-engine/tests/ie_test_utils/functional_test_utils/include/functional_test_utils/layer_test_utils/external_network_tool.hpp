// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <fstream>

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
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

protected:
    ExternalNetworkTool() = default;

    ~ExternalNetworkTool() = default;

public:
    void dumpNetworkToFile(const std::shared_ptr<ngraph::Function>& network,
                           std::string network_name) const;

    InferenceEngine::CNNNetwork loadNetworkFromFile(const std::shared_ptr<InferenceEngine::Core> core,
                                                    std::string network_name) const;

    static ExternalNetworkTool &getInstance();

    static std::string getModelsPath() { return std::string(modelsPath); }

    static ExternalNetworkMode getMode() { return mode; }

    static void setModelsPath(std::string &val) { modelsPath = val.c_str(); }

    static void setMode(ExternalNetworkMode val) { mode = val; }
};

}  // namespace LayerTestsUtils