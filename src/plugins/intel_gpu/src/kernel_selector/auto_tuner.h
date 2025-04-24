// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <mutex>
#include <map>
#include <string>
#include <memory>
#include <tuple>

#include "kernel_selector_common.h"
#include "kernel_selector_params.h"

namespace kernel_selector {

class TuningCache {
public:
    using Entry = std::tuple<std::string, int>;

    // Reads tuning cache from file.
    // Additionally the constructor may modify the internal representation to update the format to newest version,
    // which may necessitate saving afterwards.
    // This class is not thread-safe and all concurrent modifications should be synchronized by owner.
    // cacheFilePath - Path to cache file
    // If file doesn't exist and createMode is false this constructor will throw.
    explicit TuningCache(const std::string& cacheFilePath);

    // Constructs empty tuning cache.
    TuningCache();

    // Returns cached kernel for specified params. If "update" moves it to newest version if found, which may require saving afterwards.
    Entry LoadKernel(const Params& params);
    // Overrides the compute units count in params.
    Entry LoadKernel(const Params& params, uint32_t computeUnitsCount);

    static TuningCache* get();

private:
    Entry LoadKernel_v1(const Params& params, uint32_t computeUnitsCount);
    Entry LoadKernel_v2(const Params& params, uint32_t computeUnitsCount);

    class Impl;
    std::shared_ptr<Impl> impl;

    static constexpr const char* version1Marker = "version_1";
    static constexpr const char* version2Marker = "version_2";
};

class AutoTuner {
public:
    AutoTuner() = default;
    std::tuple<std::string, int> LoadKernelOffline(const Params& params);

private:
    std::mutex mutex;  // Mutex to synchronize cache updates

    /*
            The offline cache contains for each hash (that is based on the node params) the best kernel/config per
       device id. This cache can be ignored by setting ENABLE_OFFLINE_TUNING_CACHE to 0 in kernel_selector.cpp (in this
       case the default path will be chosen). Follow these steps in order to change the data inside this cache:
            1. Find the proper device ID entry.
               For example: 0x193B for SKL GT4.
               This device ID is also written in the cache file that is generated in the on-line mode.
            2. Find the hash of the node you want to change.
               This hash can be obtained by:
               std::string hash = std::to_string(create_hash(params.to_string()));
            3. Change the kernel name and/or config index.
               For example:
               { "17001023283013862129", std::make_tuple("convolution_gpu_bfyx_os_iyx_osv16", 203) }
               Means:
               For hash 17001023283013862129 the best kernel name is convolution_gpu_bfyx_os_iyx_osv16 and the config
       index is 203. If the config index is -1, it means that this is the default config for this kernel. If there are
       more configs (for example for convolution_gpu_bfyx_os_iyx_osv16 kernel) you need to find the proper config index.
               For example, for the convolution_gpu_bfyx_os_iyx_osv16 kernel you need to take a look in the constructor
       (ConvolutionKernel_bfyx_os_iyx_osv16::ConvolutionKernel_bfyx_os_iyx_osv16) – this is the index in the
       autoTuneOptions array.
        */
};
}  // namespace kernel_selector
