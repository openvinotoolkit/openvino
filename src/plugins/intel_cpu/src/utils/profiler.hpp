// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#ifdef OV_CPU_WITH_PROFILER

#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

extern "C" {
#ifdef _WIN32
#    include <intrin.h>
#else
#    include <x86intrin.h>
#endif
}

namespace ov {
namespace intel_cpu {

struct ProfileData {
    uint64_t start;
    uint64_t end;
    std::string cat;
    std::string name;

    ProfileData(const std::string& cat, const std::string& name) : cat(cat), name(name) {
        start = __rdtsc();
    }
    static void record_end(ProfileData* p);
};

typedef std::unique_ptr<ProfileData, void (*)(ProfileData*)> ProfileDataHandle;

class chromeTrace;
struct ProfilerManagerFinalizer;

class ProfilerManager {
    bool enabled;
    std::deque<ProfileData> all_data;
    std::thread::id tid;
    int serial;

public:
    ProfilerManager();
    ~ProfilerManager();

    ProfileDataHandle startProfile(const std::string& cat, const std::string& name = {}) {
        if (!enabled) {
            return ProfileDataHandle(nullptr, [](ProfileData*) {});
        }
        all_data.emplace_back(cat, name);
        return ProfileDataHandle(&all_data.back(), ProfileData::record_end);
    }

    friend class ProfilerManagerFinalizer;
};

extern thread_local ProfilerManager profilerManagerInstance;

}  // namespace intel_cpu
}  // namespace ov

#    define PROFILE(var_name, ...)                                         \
        auto var_name = ov::intel_cpu::profilerManagerInstance.startProfile(__VA_ARGS__); \
        (void)var_name;

#    define PROFILE_NEXT(var_name, ...)                                         \
        var_name = ov::intel_cpu::profilerManagerInstance.startProfile(__VA_ARGS__);

#else

#    define PROFILE(var_name, ...)
#    define PROFILE_NEXT(var_name, ...)

#endif