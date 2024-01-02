// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#if defined(OPENVINO_ARCH_X86_64)

#    include <atomic>
#    include <chrono>
#    include <cstddef>
#    include <deque>
#    include <fstream>
#    include <functional>
#    include <iostream>
#    include <map>
#    include <memory>
#    include <mutex>
#    include <set>
#    include <sstream>
#    include <thread>
#    include <vector>
extern "C" {
#    ifdef _WIN32
#        include <intrin.h>
#    else
#        include <x86intrin.h>
#    endif
}

namespace ov {
namespace intel_cpu {

namespace detail {
struct ProfileData {
    uint64_t start;
    uint64_t end;
    std::string cat;
    std::string name;

    ProfileData(const std::string& cat, const std::string& name) : cat(cat), name(name) {
        start = __rdtsc();
    }
    static void record_end(ProfileData* p) {
        p->end = __rdtsc();
    }
};

struct chromeTrace {
    std::ostream& os;
    int fake_tid;
    uint64_t ts;
    chromeTrace(std::ostream& os, int fake_tid) : os(os), fake_tid(fake_tid) {}
    void addCompleteEvent(std::string name, std::string cat, double start, double dur) {
        // chrome tracing will show & group-by to name, so we use cat as name
        os << "{\"ph\": \"X\", \"name\": \"" << cat << "\", \"cat\":\"" << name << "\","
           << "\"pid\": " << fake_tid << ", \"tid\": 0,"
           << "\"ts\": " << start << ", \"dur\": " << dur << "},\n";
    }
};

struct TscCounter {
    uint64_t tsc_ticks_per_second;
    uint64_t tsc_ticks_base;
    double tsc_to_usec(uint64_t tsc_ticks) const {
        return (tsc_ticks - tsc_ticks_base) * 1000000.0 / tsc_ticks_per_second;
    }
    double tsc_to_usec(uint64_t tsc_ticks0, uint64_t tsc_ticks1) const {
        return (tsc_ticks1 - tsc_ticks0) * 1000000.0 / tsc_ticks_per_second;
    }
    TscCounter() {
        static std::once_flag flag;
        std::call_once(flag, [&]() {
            uint64_t start_ticks = __rdtsc();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            tsc_ticks_per_second = (__rdtsc() - start_ticks);
            std::cout << "[OV_CPU_PROFILE] tsc_ticks_per_second = " << tsc_ticks_per_second << std::endl;
            tsc_ticks_base = __rdtsc();
        });
    }
};

class ProfilerBase {
public:
    virtual void save(std::ofstream& fw, TscCounter& tsc) = 0;
};

struct ProfilerFinalizer {
    std::mutex g_mutex;
    std::set<ProfilerBase*> all_managers;
    const char* dump_file_name = "ov_profile.json";
    bool dump_file_over = false;
    bool not_finalized = true;
    std::ofstream fw;
    std::atomic_int totalProfilerManagers{0};
    TscCounter tsc;

    ~ProfilerFinalizer() {
        if (not_finalized)
            finalize();
    }

    void finalize() {
        if (!not_finalized)
            return;
        std::lock_guard<std::mutex> guard(g_mutex);
        if (dump_file_over || all_managers.empty())
            return;

        // start dump
        fw.open(dump_file_name, std::ios::out);
        fw << "{\n";
        fw << "\"schemaVersion\": 1,\n";
        fw << "\"traceEvents\": [\n";
        fw.flush();

        for (auto& pthis : all_managers) {
            pthis->save(fw, tsc);
        }
        all_managers.clear();

        fw << R"({
            "name": "Profiler End",
            "ph": "i",
            "s": "g",
            "pid": "Traces",
            "tid": "Trace OV Profiler",
            "ts":)"
           << tsc.tsc_to_usec(__rdtsc()) << "}",
            fw << "]\n";
        fw << "}\n";
        auto total_size = fw.tellp();
        fw.close();
        dump_file_over = true;
        not_finalized = false;
        std::cout << "[OV_CPU_PROFILE] Dumpped " << total_size / (1024 * 1024) << " (MB) to " << dump_file_name
                  << std::endl;
    }

    int register_manager(ProfilerBase* pthis) {
        std::lock_guard<std::mutex> guard(g_mutex);
        std::stringstream ss;
        auto serial_id = totalProfilerManagers.fetch_add(1);
        ss << "[OV_CPU_PROFILE] #" << serial_id << "(" << pthis << ") : is registed." << std::endl;
        std::cout << ss.str();
        all_managers.emplace(pthis);
        return serial_id;
    }

    static ProfilerFinalizer& get() {
        static ProfilerFinalizer inst;
        return inst;
    }
};

}  // namespace detail

class Profiler : public detail::ProfilerBase {
    bool enabled;
    std::deque<detail::ProfileData> all_data;
    int serial;

public:
    Profiler() {
        const char* str_enable = std::getenv("OV_CPU_PROFILE");
        if (!str_enable)
            str_enable = "0";
        enabled = atoi(str_enable) > 0;
        if (enabled)
            serial = detail::ProfilerFinalizer::get().register_manager(this);
    }
    ~Profiler() {
        detail::ProfilerFinalizer::get().finalize();
    }

    void save(std::ofstream& fw, detail::TscCounter& tsc) override {
        if (!enabled)
            return;
        auto data_size = all_data.size();
        if (!data_size)
            return;

        detail::chromeTrace ct(fw, serial);
        for (auto& d : all_data) {
            ct.addCompleteEvent(d.name,
                                d.cat,
                                tsc.tsc_to_usec(d.start),
                                tsc.tsc_to_usec(d.start, d.end));
        }
        all_data.clear();
        std::cout << "[OV_CPU_PROFILE] #" << serial << "(" << this << ") finalize: dumpped " << data_size << std::endl;
    }

    using ProfileDataHandle = std::unique_ptr<detail::ProfileData, void (*)(detail::ProfileData*)>;

    static ProfileDataHandle startProfile(const std::string& cat, const std::string& name = {}) {
        thread_local Profiler inst;
        if (!inst.enabled) {
            return ProfileDataHandle(nullptr, [](detail::ProfileData*) {});
        }
        inst.all_data.emplace_back(cat, name);
        return ProfileDataHandle(&inst.all_data.back(), detail::ProfileData::record_end);
    }

    friend class ProfilerFinalizer;
};

}  // namespace intel_cpu
}  // namespace ov

#    define PROFILE(var_name, ...)                                          \
        auto var_name = ov::intel_cpu::Profiler::startProfile(__VA_ARGS__); \
        (void)var_name;

#    define PROFILE2(var_name, ...) var_name = ov::intel_cpu::Profiler::startProfile(__VA_ARGS__);

#else

#    define PROFILE(var_name, ...)
#    define PROFILE2(var_name, ...)

#endif