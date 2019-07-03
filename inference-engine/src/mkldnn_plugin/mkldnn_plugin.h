// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_graph.h"
#include <string>
#include <map>
#include <unordered_map>
#include <memory>
#include <functional>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>

namespace MKLDNNPlugin {

class SimpleDataHash {
public:
    SimpleDataHash() {
        for (int i = 0; i < kTableSize; i++) {
            uint64_t c = i;
            for (int j = 0; j < 8; j++)
                c = ((c & 1) ? 0xc96c5795d7870f42 : 0) ^ (c >> 1);
            table[i] = c;
        }
    }
    // Computes 64-bit "cyclic redundancy check" sum, as specified in ECMA-182
    uint64_t hash(const unsigned char* data, size_t size) const {
        uint64_t crc = 0;
        for (size_t idx = 0; idx < size; idx++)
            crc = table[(unsigned char)crc ^ data[idx]] ^ (crc >> 8);

        return ~crc;
    }

protected:
    static const int kTableSize = 256;
    uint64_t table[kTableSize];
};

class MKLDNNWeightsSharing {
public:
    MKLDNNMemoryPtr findOrCreate(const std::string& name_hash,
                             std::function<MKLDNNMemoryPtr(void)> create) {
        std::unique_lock<std::mutex> lock(guard);
        auto found = sharedWeights.find(name_hash);

        MKLDNNMemoryPtr ptr;
        if (found == sharedWeights.end() || !(ptr = found->second.lock())) {
            ptr = create();
            sharedWeights[name_hash] = ptr;
        }
        return ptr;
    }
    static const SimpleDataHash& GetHashFunc () { return simpleCRC; }

protected:
    std::unordered_map<std::string, std::weak_ptr<MKLDNNMemory>> sharedWeights;
    std::mutex guard;
    static const SimpleDataHash simpleCRC;
};

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    Engine() = default;
    ~Engine() override = default;

    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(InferenceEngine::ICNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;

    void AddExtension(InferenceEngine::IExtensionPtr extension) override;
    /**
     * @deprecated
     * @param config
     */
    void SetConfig(const std::map<std::string, std::string> &config) override;

    /**
     * @deprecated Use the version with config parameter
     */
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network, InferenceEngine::QueryNetworkResult& res) const override;
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network,
                      const std::map<std::string, std::string>& config, InferenceEngine::QueryNetworkResult& res) const override;

    static MKLDNNWeightsSharing& GetWeightsSharing() { return weightsSharing; }

private:
    Config engConfig;
    MKLDNNExtensionManager::Ptr extensionManager = std::make_shared<MKLDNNExtensionManager>();

protected:
    static MKLDNNWeightsSharing weightsSharing;
};

}  // namespace MKLDNNPlugin
