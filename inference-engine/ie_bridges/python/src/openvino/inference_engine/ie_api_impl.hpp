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

    struct IEExecNetworkPython : public InferenceEngineBridge::IEExecNetwork {
        IEExecNetworkPython(const std::string &name, std::size_t num_requests);

        PyObject *getMetric(const std::string &metric_name);

        PyObject *getConfig(const std::string &metric_name);
    };


    struct IECorePython : public InferenceEngineBridge::IECore {

        explicit IECorePython(const std::string &xmlConfigFile = std::string());

        PyObject *getMetric(const std::string &deviceName, const std::string &name);

        PyObject *getConfig(const std::string &deviceName, const std::string &name);


    };
}