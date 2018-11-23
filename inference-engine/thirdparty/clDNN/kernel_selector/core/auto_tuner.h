/*
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
*/

#pragma once

#include <atomic>
#include <mutex>
#include <map>
#include "kernel_selector_common.h"

namespace kernel_selector 
{
    struct tuning_data // this could be replaced with 
    {
        std::map<std::string, std::tuple<std::string, int>> td;
    };

    class AutoTuner
    {
    public:
        AutoTuner() = default;
        std::tuple<std::string, int> LoadKernelOnline(const TuningMode tuningMode, const std::string& tuningFilePath, const std::string& deviceID, const std::string& driverVersion, const std::string& hostVersion, const std::string& hash);
        void StoreKernel(const std::string& tuningFilePath, const std::string& hash, const std::string& implementationName, const int tuneIndex);
        std::tuple<std::string, int> LoadKernelOffline(const std::string& deviceID, const std::string& hash);

    private:    
        std::map<std::string, tuning_data> onlineCache; // Tuning file name -> kernel/config per hash (hash -> [implementation name, tuning index])
        std::mutex mutex; // Mutex to synchronize cache updates
        
        /*
            The offline cache contains for each hash (that is based on the node params) the best kernel/config per device id.
            This cache can be ignored by setting ENABLE_OFFLINE_TUNING_CACHE to 0 in kernel_selector.cpp (in this case the default path will be chosen).
            Follow these steps in order to change the data inside this cache:
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
               For hash 17001023283013862129 the best kernel name is convolution_gpu_bfyx_os_iyx_osv16 and the config index is 203.
               If the config index is -1, it means that this is the default config for this kernel.
               If there are more configs (for example for convolution_gpu_bfyx_os_iyx_osv16 kernel) you need to find the proper config index.
               For example, for the convolution_gpu_bfyx_os_iyx_osv16 kernel you need to take a look in the constructor (ConvolutionKernel_bfyx_os_iyx_osv16::ConvolutionKernel_bfyx_os_iyx_osv16) – 
               this is the index in the autoTuneOptions array.
        */
    };
}