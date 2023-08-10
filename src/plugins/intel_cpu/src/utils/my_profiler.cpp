// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "my_profiler.hpp"

#ifdef WIN32
#include <intrin.h>
#pragma intrinsic(__rdtsc)
#else
#include <x86intrin.h>
#endif

struct dump_items {
    std::string name;       // The name of the event, as displayed in Trace Viewer
    std::string cat;        // The event categories
    std::string ph = "X";   // The event type, 'B'/'E' OR 'X'
    std::string pid = "0";  // The process ID for the process
    std::string tid;        // The thread ID for the thread that output this event.
    uint64_t ts1;           // The tracing clock timestamp of the event, [microsecond]
    uint64_t ts2;           // Duration = ts2 - ts1.
    std::string tts;        // Optional. The thread clock timestamp of the event
    std::vector<std::pair<std::string, std::string>> vecArgs;
};

static inline std::string get_thread_id() {
    std::stringstream ss;
    ss << std::this_thread::get_id();
    return ss.str();
}

static uint64_t rdtsc_calibrate(int seconds = 1) {
    uint64_t start_ticks = __rdtsc();
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    return (__rdtsc() - start_ticks) / seconds;
}

class ProfilerManager {
protected:
    std::vector<dump_items> _vecItems;
    std::atomic<uint64_t> tsc_ticks_per_second{0};
    std::atomic<uint64_t> tsc_ticks_base{0};
    std::mutex _mutex;

public:
    ProfilerManager() {
        if (tsc_ticks_per_second == 0) {
            uint64_t expected = 0;
            auto tps = rdtsc_calibrate();
            tsc_ticks_per_second.compare_exchange_strong(expected, tps);
            std::cout << "=== ProfilerManager: tsc_ticks_per_second = " << tsc_ticks_per_second << std::endl;
            tsc_ticks_base.compare_exchange_strong(expected, __rdtsc());
            std::cout << "=== ProfilerManager: tsc_ticks_base = " << tsc_ticks_base << std::endl;
        }
    }

    ProfilerManager(ProfilerManager& other) = delete;
    void operator=(const ProfilerManager&) = delete;
    ~ProfilerManager() {
        // Save tracing log to json file.
        printf("ProfilerManager unconstruct...............\n");
        save_to_json();
    }

    void add(const dump_items& val) {
        std::lock_guard<std::mutex> lk(_mutex);
        _vecItems.emplace_back(val);
    }

private:
    std::string tsc_to_nsec(uint64_t tsc_ticks) {
        double val = (tsc_ticks - tsc_ticks_base) * 1000000.0 / tsc_ticks_per_second;
        return std::to_string(val);
    }

    std::string tsc_to_nsec(uint64_t start, uint64_t end) {
        double val = (end - start) * 1000000.0 / tsc_ticks_per_second;
        return std::to_string(val);
    }

    void save_to_json() {
        std::string out_fn = std::string("profile_cpu_") + std::to_string(tsc_ticks_base) + ".json";
        FILE* pf = fopen(out_fn.c_str(), "wb");
        if (nullptr == pf) {
            printf("Can't fopen: %s\n", out_fn.c_str());
            return;
        }
        printf("Save profile log: %s\n", out_fn.c_str());

        // Headers
        fprintf(pf, "{\n\"schemaVersion\": 1,\n\"traceEvents\":[\n");

        for (size_t i = 0; i < _vecItems.size(); i++) {
            auto& itm = _vecItems[i];
            // Write 1 event
            fprintf(pf, "{");
            fprintf(pf, "\"name\":\"%s\",", itm.name.c_str());
            fprintf(pf, "\"cat\":\"%s\",", itm.cat.c_str());
            fprintf(pf, "\"ph\":\"%s\",", itm.ph.c_str());
            fprintf(pf, "\"pid\":\"%s\",", itm.pid.c_str());
            fprintf(pf, "\"tid\":\"%s\",", itm.tid.c_str());
            fprintf(pf, "\"ts\":\"%s\",", tsc_to_nsec(itm.ts1).c_str());
            fprintf(pf, "\"dur\":\"%s\",", tsc_to_nsec(itm.ts1, itm.ts2).c_str());
            fprintf(pf, "\"args\":{");
            for (size_t j = 0; j < itm.vecArgs.size(); j++) {
                fprintf(pf,
                        "\"%s\":\"%s\"%s",
                        itm.vecArgs[j].first.c_str(),
                        itm.vecArgs[j].second.c_str(),
                        j + 1 == itm.vecArgs.size() ? "" : ",");
            }
            fprintf(pf, "}}%s\n", i == _vecItems.size() - 1 ? "" : ",");
        }

        fprintf(pf, "]\n}\n");
        fclose(pf);
    }
};
static ProfilerManager g_profileManage;
MyProfile::MyProfile(const std::string& name, const std::vector<std::pair<std::string, std::string>>& args) {
    _name = name;
    _args = args;
    _ts1 = __rdtsc();
}

MyProfile::~MyProfile() {
    dump_items itm;
    itm.ts2 = __rdtsc();
    itm.ts1 = _ts1;
    itm.name = _name;
    itm.tid = get_thread_id();
    itm.cat = "PERF";
    itm.vecArgs = _args;
    g_profileManage.add(itm);
}