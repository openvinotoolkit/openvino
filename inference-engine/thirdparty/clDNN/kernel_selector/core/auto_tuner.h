// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once

#include <atomic>
#include <mutex>
#include <map>
#include <string>
#include "kernel_selector_common.h"
#include "kernel_selector_params.h"
#include "document.h"
#include <memory>
#include <tuple>

namespace kernel_selector {

class TuningCache {
public:
    using Entry = std::tuple<std::string, int>;

    // Reads tuning cache from file.
    // Additionally the constructor may modify the internal representation to update the format to newest version,
    // which may necessitate saving afterwards.
    // This class is not thread-safe and all concurrent modifications should be synchronized by owner.
    // cacheFilePath - Path to cache file
    // createMode    - Flag to enable creation if cache file does not exist.
    // If file doesn't exist and createMode is false this constructor will throw.
    explicit TuningCache(const std::string& cacheFilePath, bool createMode = false);

    // Constructs empty tuning cache.
    TuningCache();

    // Returns cached kernel for specified params. If "update" moves it to newest version if found, which may require saving afterwards.
    Entry LoadKernel(const Params& params, bool update = true);
    // Overrides the compute units count in params.
    Entry LoadKernel(const Params& params, uint32_t computeUnitsCount, bool update = true);
    // Stores kernel for specified params.
    void StoreKernel(const Params& params, const std::string& implementationName, int tuneIndex);
    // Overrides the compute units count in params.
    void StoreKernel(const Params& params, uint32_t computeUnitsCount, const std::string& implementationName, int tuneIndex);
    // Removes the cached kernel for specified params if it exists, for all cache versions.
    void RemoveKernel(const Params& params);
    // Saves the internal cache to specified file.
    void Save(const std::string& cacheFilePath);

    bool NeedsSave() const { return needsSave; }

private:
    Entry LoadKernel_v1(const Params& params, uint32_t computeUnitsCount);
    Entry LoadKernel_v2(const Params& params, uint32_t computeUnitsCount);

    bool RemoveKernel_v1(const Params& params, uint32_t computeUnitsCount);
    bool RemoveKernel_v2(const Params& params, uint32_t computeUnitsCount);


    rapidjson::Document cache;
    bool needsSave;

    static constexpr const char* version1Marker = "version_1";
    static constexpr const char* version2Marker = "version_2";
};

class AutoTuner {
public:
    AutoTuner() = default;
    std::tuple<std::string, int> LoadKernelOnline(const TuningMode tuningMode,
                                                  const std::string& cacheFilePath,
                                                  const Params& params);
    void StoreKernel(const std::string& cacheFilePath,
                     const Params& params,
                     std::string implementationName,
                     const int tuneIndex);
    void RemoveKernel(const std::string& cacheFilePath,
                      const Params& params);
    std::tuple<std::string, int> LoadKernelOffline(std::shared_ptr<TuningCache> cache,
                                                   const Params& params);

private:
    std::string lastCachePath;
    std::shared_ptr<TuningCache> onlineCache;
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
