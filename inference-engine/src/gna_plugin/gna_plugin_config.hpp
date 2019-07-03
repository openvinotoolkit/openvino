// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include <memory>
#include <utility>
#include <ie_icnn_network.hpp>
#include "ie_common.h"
#include "gna_plugin_log.hpp"

namespace GNAPluginNS {

using CNNNetworkPtr = std::shared_ptr<InferenceEngine::ICNNNetwork>;

struct Endpoint {
    InferenceEngine::TargetDevice device;
    InferenceEngine::Precision networkPrec;
    std::function<CNNNetworkPtr(InferenceEngine::ICNNNetwork &network)> convert;

    Endpoint(InferenceEngine::TargetDevice device,
             InferenceEngine::Precision networkPrec,
             std::function<CNNNetworkPtr(InferenceEngine::ICNNNetwork &network)> converter = [](InferenceEngine::ICNNNetwork &network) {
                 return CNNNetworkPtr(&network, [](InferenceEngine::ICNNNetwork *nodelete) {});
             }) : device(device), networkPrec(networkPrec), convert(converter) {
    }
};

class Config {
 public:
    using Desc = std::vector<Endpoint>;
    Desc supported;
    InferenceEngine::TargetDevice _defaultDevice = InferenceEngine::TargetDevice::eDefault;

 public:
    explicit Config(std::vector<Endpoint> &&config)
        : supported(std::move(config)) {
    }

    /**
     * @brief default device value is plugin dependent, so it should be also set, to allow fallback
     */
    void setDefaultDevice(InferenceEngine::TargetDevice d) {
        _defaultDevice = d;
    }

    inline Endpoint find_configuration(InferenceEngine::ICNNNetwork &network) {
        auto device = network.getTargetDevice();
        auto targetDevice = device == InferenceEngine::TargetDevice::eDefault ? _defaultDevice : device;

        auto res = std::find_if(std::begin(supported), std::end(supported), [&](Endpoint &e) {
            return e.networkPrec == network.getPrecision() && (
                e.device == device ||
                    e.device == targetDevice);
        });

        if (res == std::end(supported)) {
            THROW_GNA_EXCEPTION << "\"The plugin doesn't support target device: "
                               << InferenceEngine::TargetDeviceInfo::name(network.getTargetDevice())
                               << ".\nSupported target device: " << InferenceEngine::TargetDeviceInfo::name(InferenceEngine::TargetDevice::eGNA);
        }

        return *res;
    }
};
}  // namespace GNAPluginNS
