// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"
#include "config.h"
#include "dnnl_scratch_pad.h"
#include "extension_mngr.h"
#include "weights_cache.hpp"

namespace ov {
namespace intel_cpu {

class GraphContext {
public:
    typedef std::shared_ptr<GraphContext> Ptr;

    GraphContext(const Config& config,
                 ExtensionManager::Ptr extensionManager,
                 WeightsSharing::Ptr w_cache,
                 std::shared_ptr<std::mutex> sharedMutex)
        : config(config),
          extensionManager(extensionManager),
          sharedMutex(sharedMutex) {
        // disable weights caching if graph was created only once
        weightsCache = config.streamExecutorConfig._streams != 1 ? w_cache : nullptr;

        rtParamsCache = std::make_shared<MultiCache>(config.rtCacheCapacity);
        rtScratchPad = std::make_shared<DnnlScratchPad>(eng);
    }

    Config& getConfig() {
        return config;
    }

    const Config& getConfig() const {
        return config;
    }

    ExtensionManager::Ptr getExtensionManager() const {
        return extensionManager;
    }

    WeightsSharing::Ptr getWeightsCache() const {
        return weightsCache;
    }

    std::shared_ptr<std::mutex> getSharedMutex() const {
        return sharedMutex;
    }

    MultiCachePtr getParamsCache() const {
        return rtParamsCache;
    }

    DnnlScratchPadPtr getScratchPad() const {
        return rtScratchPad;
    }

    dnnl::engine getEngine() const {
        return eng;
    }

    bool getGraphQuantizedFlag() const {
        return isGraphQuantized;
    }

    void setGraphQuantizedFlag(bool on) {
        isGraphQuantized = on;
    }

private:
    Config config;  // network-level config

    ExtensionManager::Ptr extensionManager;
    WeightsSharing::Ptr weightsCache;         // per NUMA node caches for sharing weights data
    std::shared_ptr<std::mutex> sharedMutex;  // mutex for protection of type-relaxed Op in clone_model()

    MultiCachePtr rtParamsCache;     // primitive cache
    DnnlScratchPadPtr rtScratchPad;  // scratch pad

    bool isGraphQuantized = false;
    static dnnl::engine eng;  // onednn engine (singleton)
};

}  // namespace intel_cpu
}  // namespace ov
