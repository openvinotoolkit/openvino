// Copyright (c) 2019 Intel Corporation
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

#include "device_cache_reader.h"
#include "include/to_string_utils.h"
#include "auto_tuner.h"
#include <limits>
#include "istreamwrapper.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
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

namespace cldnn {
namespace gpu {
namespace {

std::shared_ptr<kernel_selector::TuningCache> get_cache_from_file(std::string tuning_cache_path) {
    if (tuning_cache_path.compare("cache.json") == 0) {
#ifdef _WIN32
        char path[MAX_PATH];
        HMODULE hm = NULL;
        GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR)&get_cache_from_file,
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
}  // namespace

device_cache_reader::device_cache_reader(const std::string tuning_file_path) {
    {
        try {
            _dev_cache = get_cache_from_file(tuning_file_path);
        }
        catch (...) {
            std::cout << "[WARNING] error during parsing cache file, tuning data won't be used" << std::endl;
            _dev_cache = std::make_shared<kernel_selector::TuningCache>();
        }
    }
}

}  // namespace gpu
}  // namespace cldnn
