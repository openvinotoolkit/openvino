// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "extension_mngr.h"

#include <iostream>
#include <functional>
#include <cpp/ie_cnn_network.h>

namespace ov {
namespace intel_cpu {

class CNNNetworkSerializer {
public:
    CNNNetworkSerializer(std::ostream & ostream, ExtensionManager::Ptr extensionManager);
    void operator << (const InferenceEngine::CNNNetwork & network);

private:
    std::ostream & _ostream;
    ExtensionManager::Ptr _extensionManager;
};

class CNNNetworkDeserializer {
public:
    typedef std::function<
                InferenceEngine::CNNNetwork(
                        const std::string&,
                        const InferenceEngine::Blob::CPtr&)> cnn_network_builder;
    CNNNetworkDeserializer(std::istream & istream, cnn_network_builder fn);
    void operator >> (InferenceEngine::CNNNetwork & network);

private:
    std::istream & _istream;
    cnn_network_builder _cnn_network_builder;
};

// const std::string& model, const Blob::CPtr& weights

}   // namespace intel_cpu
}   // namespace ov
