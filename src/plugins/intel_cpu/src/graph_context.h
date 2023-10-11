// Copyright (C) 2018-2023 Intel Corporation
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
                 bool isGraphQuantized)
        : config(config),
          extensionManager(extensionManager),
          weightsCache(w_cache),
          isGraphQuantizedFlag(isGraphQuantized) {
        rtParamsCache = std::make_shared<MultiCache>(config.rtCacheCapacity);
        rtScratchPad = std::make_shared<DnnlScratchPad>(getEngine());
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


    MultiCachePtr getParamsCache() const {
        return rtParamsCache;
    }

    DnnlScratchPadPtr getScratchPad() const {
        return rtScratchPad;
    }

    static const dnnl::engine& getEngine();

    bool isGraphQuantized() const {
        return isGraphQuantizedFlag;
    }

private:
    Config config;  // network-level config

    ExtensionManager::Ptr extensionManager;
    WeightsSharing::Ptr weightsCache;         // per NUMA node caches for sharing weights data

    MultiCachePtr rtParamsCache;     // primitive cache
    DnnlScratchPadPtr rtScratchPad;  // scratch pad

    bool isGraphQuantizedFlag = false;
};

}  // namespace intel_cpu
}  // namespace ov
