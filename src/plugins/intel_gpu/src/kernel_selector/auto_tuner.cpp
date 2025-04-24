// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_tuner.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include <memory>
#include <utility>
#include <tuple>

#include "rapidjson/istreamwrapper.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/document.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <setupapi.h>
#include <devguid.h>
#include <cstring>
#else
#include <unistd.h>
#include <limits.h>
#include <link.h>
#include <dlfcn.h>
#endif

#if __cplusplus > 201703L

// Add operators `==` and `!=` for rapidjson::GenericMemberIterator for non const iterator when build with C++20,
// is more strict regarding type checks.
namespace rapidjson {

template <typename Encoding, typename Allocator>
inline bool operator==(GenericMemberIterator<false, Encoding, Allocator> lhs,
                       GenericMemberIterator<false, Encoding, Allocator> rhs) {
    return static_cast<GenericMemberIterator<true, Encoding, Allocator>>(lhs) ==
           static_cast<GenericMemberIterator<true, Encoding, Allocator>>(rhs);
}

template <typename Encoding, typename Allocator>
inline bool operator!=(GenericMemberIterator<false, Encoding, Allocator> lhs,
                       GenericMemberIterator<false, Encoding, Allocator> rhs) {
    return !(lhs == rhs);
}
}  // namespace rapidjson
#endif

namespace kernel_selector {

class TuningCache::Impl {
public:
    rapidjson::Document cache;
};

TuningCache::TuningCache(const std::string& cacheFilePath)
    : impl(new Impl()) {
    // Read cache file
    std::ifstream tuningFile(cacheFilePath);

    if (tuningFile && tuningFile.good()) {
        std::stringstream buffer;
        buffer << tuningFile.rdbuf();
        impl->cache.Parse(buffer.str().c_str());
    } else {
        throw std::runtime_error("Tuning file: " + cacheFilePath + " could not be read! Must provide a valid cache file in USE_CACHE mode.");
    }

    if (impl->cache.IsNull()) {
        impl->cache.SetObject();
    } else if (!impl->cache.IsObject()) {
        throw std::runtime_error("Tuning file: " + cacheFilePath + " has incorrect format.");
    }

    auto cacheObj = impl->cache.GetObject();

    // Update to new format with version markers
    if (!cacheObj.HasMember(version2Marker)) {
        auto newName = rapidjson::Value(version2Marker, impl->cache.GetAllocator());
        auto newObj = rapidjson::Value(rapidjson::Type::kObjectType);
        cacheObj.AddMember(newName, newObj, impl->cache.GetAllocator());
    }

    bool needsV1 = false;
    for (auto& member : cacheObj) {
        std::string nameStr = member.name.GetString();
        if (nameStr != version1Marker && nameStr != version2Marker) {
            needsV1 = true;
        }
    }

    if (needsV1) {
        if (!cacheObj.HasMember(version1Marker)) {
            auto newName = rapidjson::Value(version1Marker, impl->cache.GetAllocator());
            auto newObj = rapidjson::Value(rapidjson::Type::kObjectType);
            cacheObj.AddMember(newName, newObj, impl->cache.GetAllocator());
        }

        for (auto it = cacheObj.begin(); it != cacheObj.end();) {
            auto& member = *it;
            std::string nameStr = member.name.GetString();
            if (nameStr != version1Marker && nameStr != version2Marker) {
                auto newName = rapidjson::Value(rapidjson::Type::kStringType);
                auto newValue = rapidjson::Value(rapidjson::Type::kObjectType);
                newName.Swap(member.name);
                newValue.Swap(member.value);
                impl->cache[version1Marker].AddMember(newName, newValue, impl->cache.GetAllocator());
                it = cacheObj.EraseMember(it);
            } else {
                it++;
            }
        }
    }
}

TuningCache::TuningCache()
    : impl(new Impl()) {
    impl->cache.SetObject();
    auto v2Name = rapidjson::Value(version2Marker, impl->cache.GetAllocator());
    auto v2Obj = rapidjson::Value(rapidjson::Type::kObjectType);
    impl->cache.AddMember(v2Name, v2Obj, impl->cache.GetAllocator());
}

TuningCache::Entry TuningCache::LoadKernel(const Params& params) {
    return LoadKernel(params, params.engineInfo.computeUnitsCount);
}

TuningCache::Entry TuningCache::LoadKernel(const Params& params, uint32_t computeUnitsCount) {
    bool oldVersion = false;
    // Try to load from version 2
    auto result = LoadKernel_v2(params, computeUnitsCount);
    // Try to load from version 1
    if (std::get<0>(result).empty()) {
        auto result_v1 = LoadKernel_v1(params, computeUnitsCount);
        oldVersion = !std::get<0>(result_v1).empty();
        if (oldVersion && std::get<0>(result).empty()) {
            result = result_v1;
        }
    }

    return result;
}

TuningCache::Entry TuningCache::LoadKernel_v1(const Params& params, uint32_t computeUnitsCount) {
    Entry result = std::make_tuple<std::string, int>("", 0);

    auto hashStr = std::to_string(create_hash(params.to_string()));
    auto computeUnitsStr = std::to_string(computeUnitsCount);

    auto v1It = impl->cache.FindMember(version1Marker);
    if (v1It == impl->cache.MemberEnd())
        return result;

    auto computeUnitsIt = v1It->value.FindMember(computeUnitsStr.c_str());
    if (computeUnitsIt == v1It->value.MemberEnd())
        return result;

    auto hashIt = computeUnitsIt->value.FindMember(hashStr.c_str());
    if (hashIt == computeUnitsIt->value.MemberEnd())
        return result;

    auto& prog = hashIt->value;
    return std::make_tuple(prog[0].GetString(), prog[1].GetInt());
}

TuningCache::Entry TuningCache::LoadKernel_v2(const Params& params, uint32_t computeUnitsCount) {
    Entry result = std::make_tuple<std::string, int>("", 0);

    auto kTypeStr = toString(params.GetType());
    auto paramStr = params.to_cache_string_v2();
    auto computeUnitsStr = std::to_string(computeUnitsCount);

    auto v2It = impl->cache.FindMember(version2Marker);
    if (v2It == impl->cache.MemberEnd())
        return result;

    auto computeUnitsIt = v2It->value.FindMember(computeUnitsStr.c_str());
    if (computeUnitsIt == v2It->value.MemberEnd())
        return result;

    auto kTypeIt = computeUnitsIt->value.FindMember(kTypeStr.c_str());
    if (kTypeIt == computeUnitsIt->value.MemberEnd())
        return result;

    auto paramIt = kTypeIt->value.FindMember(paramStr.c_str());
    if (paramIt == kTypeIt->value.MemberEnd())
        return result;

    auto& prog = paramIt->value;
    return std::make_tuple(prog[0].GetString(), prog[1].GetInt());
}

std::tuple<std::string, int> AutoTuner::LoadKernelOffline(const Params& params) {
    std::lock_guard<std::mutex> lock(mutex);
    static const uint32_t defaultComputeUnits = 24;
    TuningCache* deviceCache = TuningCache::get();
    if (!deviceCache)
        return {};
    auto result = deviceCache->LoadKernel(params);
    if (std::get<0>(result).empty() && params.engineInfo.computeUnitsCount != defaultComputeUnits) {
        result = deviceCache->LoadKernel(params, defaultComputeUnits);
    }
    return result;
}

TuningCache* TuningCache::get() {
    static std::mutex m;
    static std::shared_ptr<TuningCache> cache_instance = nullptr;
    std::lock_guard<std::mutex> lock(m);
    std::string path = "cache.json";
#ifdef _WIN32
    char module_path[MAX_PATH];
    HMODULE hm = NULL;
    GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)&TuningCache::get,
        &hm);
    GetModuleFileName(hm, module_path, sizeof(module_path));
    std::string bin_path(module_path);
    path = bin_path.substr(0, bin_path.find_last_of("\\")) + "\\cache.json";
#elif __linux__
    const char* device_info_failed_msg = "Device lookup failed";
    Dl_info dl_info;
    dladdr((void*)(device_info_failed_msg), &dl_info);  // NOLINT
    std::string bin_path(dl_info.dli_fname);
    path = bin_path.substr(0, bin_path.find_last_of("/")) + "/cache.json";
#else
#error "Intel GPU plugin: unknown target system"
#endif

    if (!cache_instance) {
        try {
            cache_instance = std::make_shared<kernel_selector::TuningCache>(path);
        } catch (...) {
            cache_instance = std::make_shared<kernel_selector::TuningCache>();
        }
    }

    return cache_instance.get();
}

}  // namespace kernel_selector
