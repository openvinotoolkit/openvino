// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"
#include "config.h"
#include "extension_mngr.h"
#include "dnnl_scratch_pad.h"
#include "weights_cache.hpp"

namespace ov {
namespace intel_cpu {

struct RuntimeEnv {
    typedef std::shared_ptr<RuntimeEnv> Ptr;

    Config config;  // network-level config

    ExtensionManager::Ptr extensionManager;
    WeightsSharing::Ptr weightsCache;         // per NUMA node caches for sharing weights data
    std::shared_ptr<std::mutex> sharedMutex;  // mutex for protection of type-relaxed Op in clone_model()

    MultiCachePtr rtParamsCache;     // primitive cache
    DnnlScratchPadPtr rtScratchPad;  // scratch pad
    static dnnl::engine eng;         // onednn engine (singleton)

    int streamId;                    // the stream ID for current runtime (so instance of RuntimeEnv is per-stream)
    int numaNodeId;                  // the NUMA node ID for current runtime

    RuntimeEnv(const Config& config,
               ExtensionManager::Ptr extensionManager,
               WeightsSharing::Ptr w_cache,
               std::shared_ptr<std::mutex> sharedMutex,
               int streamId,
               int numaNodeId)
        : config(config),
          extensionManager(extensionManager),
          sharedMutex(sharedMutex),
          streamId(streamId),
          numaNodeId(numaNodeId) {
        // disable weights caching if graph was created only once
        weightsCache = config.streamExecutorConfig._streams != 1 ? w_cache : nullptr;

        rtParamsCache = std::make_shared<MultiCache>(config.rtCacheCapacity);
        rtScratchPad = std::make_shared<DnnlScratchPad>(eng);
    }
};

}  // namespace intel_cpu
}  // namespace ov
