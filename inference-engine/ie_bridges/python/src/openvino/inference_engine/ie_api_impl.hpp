// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Python.h"

#include <iterator>
#include "ie_exec_network.h"
#include "ie_core.h"
#include "infer_request_wrapper.h"
#include "ie_plugin.h"
#include "helpers.h"


namespace InferenceEnginePython {

    struct IEExecNetwork : public InferenceEngineBridge::IEExecNetwork {
        IEExecNetwork(const std::string &name, std::size_t num_requests);

        PyObject *getMetric(const std::string &metric_name);

        PyObject *getConfig(const std::string &metric_name);

    };


    struct IECore : public InferenceEngineBridge::IECore {

        explicit IECore(const std::string &xmlConfigFile = std::string());

        PyObject *getMetric(const std::string &deviceName, const std::string &name);

        PyObject *getConfig(const std::string &deviceName, const std::string &name);

        std::unique_ptr<InferenceEnginePython::IEExecNetwork> loadNetwork(InferenceEngineBridge::IENetwork network,
                                                                          const std::string &deviceName,
                                                                          const std::map<std::string, std::string> &config,
                                                                          int &num_requests);
    };

    struct IEPlugin : public InferenceEngineBridge::IEPlugin {

        IEPlugin(const std::string &device, const std::vector<std::string> &plugin_dirs);

        IEPlugin() = default;

        std::unique_ptr<InferenceEnginePython::IEExecNetwork>
        load(const InferenceEngineBridge::IENetwork &net,
             int num_requests,
             const std::map<std::string, std::string> &config);
    };
}