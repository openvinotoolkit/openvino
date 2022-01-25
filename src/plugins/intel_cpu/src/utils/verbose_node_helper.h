// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#include "cache/cache_entry.h"

namespace MKLDNNPlugin {

struct VerboseNodeStorage {
    void cleanup() {
        prepareParamsCacheLookUpStatus = CacheEntryBase::LookUpStatus::Miss;
    }

    bool isPrepareParamsCacheHit() const {
        return prepareParamsCacheLookUpStatus == CacheEntryBase::LookUpStatus::Hit;
    }

    CacheEntryBase::LookUpStatus prepareParamsCacheLookUpStatus;
};

#define VERBOSE_HELPER_NODE_PREPARE_PARAMS(lookUpStatus) \
    _verboseStorage.prepareParamsCacheLookUpStatus = lookUpStatus
} // namespace MKLDNNPlugin
#else
#define VERBOSE_HELPER_NODE_PREPARE_PARAMS(lookUpStatus)
#endif // CPU_DEBUG_CAPS