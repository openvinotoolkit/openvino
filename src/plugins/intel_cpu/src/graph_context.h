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
    typedef std::shared_ptr<const GraphContext> CPtr;

    GraphContext(const Config& config,
                 ExtensionManager::Ptr extensionManager,
                 WeightsSharing::Ptr w_cache,
                 std::shared_ptr<std::mutex> sharedMutex,
                 bool isGraphQuantized)
        : config(config),
          extensionManager(extensionManager),
          weightsCache(w_cache),
          sharedMutex(sharedMutex),
          isGraphQuantizedFlag(isGraphQuantized) {
        rtParamsCache = std::make_shared<MultiCache>(config.rtCacheCapacity);
        rtScratchPad = std::make_shared<DnnlScratchPad>(eng);
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

    bool isGraphQuantized() const {
        return isGraphQuantizedFlag;
    }

private:
    Config config;  // network-level config

    ExtensionManager::Ptr extensionManager;
    WeightsSharing::Ptr weightsCache;         // per NUMA node caches for sharing weights data
    std::shared_ptr<std::mutex> sharedMutex;  // mutex for protection of type-relaxed Op in clone_model()

    MultiCachePtr rtParamsCache;     // primitive cache
    DnnlScratchPadPtr rtScratchPad;  // scratch pad

    bool isGraphQuantizedFlag = false;
    static dnnl::engine eng;  // onednn engine (singleton)
};

}  // namespace intel_cpu
}  // namespace ov
