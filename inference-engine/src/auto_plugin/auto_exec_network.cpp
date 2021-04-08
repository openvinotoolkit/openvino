// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>

#include "ie_metric_helpers.hpp"
#include "auto_exec_network.hpp"
#include "auto_infer_request.hpp"

namespace AutoPlugin {
    using namespace InferenceEngine;

AutoExecutableNetwork::AutoExecutableNetwork(const ExecutableNetwork& network,
                                             const DeviceInformation& deviceInfo,
                                             const bool               needPerfCounters) :
    _deviceInfo(deviceInfo),
    _network(network),
    _config(deviceInfo.config.begin(), deviceInfo.config.end()),
    _needPerfCounters(needPerfCounters) {
}

AutoExecutableNetwork::~AutoExecutableNetwork() = default;

RemoteContext::Ptr AutoExecutableNetwork::GetContext() const {
    try {
        return _network.GetContext();
    } catch (const NotImplemented&) {
        IE_THROW(NotImplemented) << "None of the devices in the AUTO has an associated remote context.";
    }
}

IInferRequestInternal::Ptr AutoExecutableNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                         OutputsDataMap networkOutputs) {
    auto inferRequest = _network.CreateInferRequest();
    return std::make_shared<AutoInferRequest>(networkInputs, networkOutputs, inferRequest);
}

IInferRequestInternal::Ptr AutoExecutableNetwork::CreateInferRequest() {
    return CreateInferRequestImpl(_networkInputs, _networkOutputs);
}

void AutoExecutableNetwork::SetConfig(const std::map<std::string, Parameter> &config) {
    IE_THROW(NotImplemented) << "Auto plugin doesn't implement SetConfig";
}

Parameter AutoExecutableNetwork::GetConfig(const std::string &name) const {
    auto it = _config.find(name);
    if (it != _config.end()) {
        return it->second;
    } else {
        IE_THROW(NotFound) << name <<" not found in the ExecutableNetwork config";
    }
}

Parameter AutoExecutableNetwork::GetMetric(const std::string &name) const {
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        unsigned int res = 0u;
            try {
              res = _network.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
            } catch (const Exception &iie) {
                  IE_THROW()
                        << "Every device used with the Auto-Device should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << _deviceInfo.deviceName << " with error:" << iie.what();
           }

        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, res);
    } else if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, _network.GetMetric(
            METRIC_KEY(NETWORK_NAME)).as<std::string>());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, {
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)
        });
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = {};
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        IE_THROW() << "Unsupported Network metric: " << name;
    }
}

}  // namespace AutoPlugin
