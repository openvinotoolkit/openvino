// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_cache_reader.h"
#include "auto_tuner.h"
#include <limits>
#include "istreamwrapper.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <SetupAPI.h>
#include <devguid.h>
#include <cstring>
#else
#include <unistd.h>
#include <limits.h>
#include <link.h>
#include <dlfcn.h>
#endif

#include <fstream>
#include <iostream>
#include <utility>

namespace kernel_selector {

std::shared_ptr<kernel_selector::TuningCache> CreateTuningCacheFromFile(std::string tuning_cache_path) {
    if (tuning_cache_path.compare("cache.json") == 0) {
#ifdef _WIN32
        char path[MAX_PATH];
        HMODULE hm = NULL;
        GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR)&CreateTuningCacheFromFile,
            &hm);
        GetModuleFileName(hm, path, sizeof(path));
        std::string bin_path(path);
        tuning_cache_path = bin_path.substr(0, bin_path.find_last_of("\\")) + "\\cache.json";
#else
        const char* device_info_failed_msg = "Device lookup failed";
        Dl_info dl_info;
        dladdr((void*)(device_info_failed_msg), &dl_info);  // NOLINT
        std::string bin_path(dl_info.dli_fname);
        tuning_cache_path = bin_path.substr(0, bin_path.find_last_of("/")) + "/cache.json";
#endif
    }

    return std::make_shared<kernel_selector::TuningCache>(tuning_cache_path, false);
}
}  // namespace kernel_selector
