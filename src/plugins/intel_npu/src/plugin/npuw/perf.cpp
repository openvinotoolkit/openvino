// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "perf.hpp"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "logging.hpp"

namespace {
std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

// Intentionally leaked: profiles may outlive any static object of this TU
std::vector<ov::npuw::perf::IProfile*>& registry() {
    static auto* r = new std::vector<ov::npuw::perf::IProfile*>();
    return *r;
}

std::vector<const void*>& boundary_owners() {
    static auto* o = new std::vector<const void*>();
    return *o;
}

// Guarded by registry_mutex(). One id per run boundary, shared by every Execution
// profile - otherwise a profile that was inactive for a whole conversation would
// number the next one differently from the others.
std::size_t g_run_index = 0u;
}  // namespace

void ov::npuw::perf::register_profile(IProfile* profile) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().push_back(profile);
}

void ov::npuw::perf::unregister_profile(IProfile* profile) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto& r = registry();
    r.erase(std::remove(r.begin(), r.end(), profile), r.end());
}

void ov::npuw::perf::reset_all(Scope scope) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    for (auto* profile : registry()) {
        if (profile->scope() == scope) {
            profile->reset();
        }
    }
}

void ov::npuw::perf::snapshot_and_reset_all(Scope scope) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    const auto ending = g_run_index++;
    for (auto* profile : registry()) {
        if (profile->scope() == scope) {
            profile->snapshot_and_reset(ending);
        }
    }
}

std::size_t ov::npuw::perf::current_run_index() {
    std::lock_guard<std::mutex> lock(registry_mutex());
    return g_run_index;
}

std::size_t ov::npuw::perf::keep_runs() {
    static const std::size_t runs = []() -> std::size_t {
        constexpr std::size_t default_runs = 8u;
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
        const auto* opt = std::getenv("OPENVINO_NPUW_PROF_KEEP_RUNS");
        if (opt == nullptr) {
            return default_runs;
        }
        try {
            const auto parsed = std::stoll(std::string(opt));
            return parsed <= 0 ? std::size_t{0u} : static_cast<std::size_t>(parsed);
        } catch (...) {
            LOG_WARN("NPUW profiling: bad OPENVINO_NPUW_PROF_KEEP_RUNS value, using " << default_runs);
        }
#endif
        return default_runs;
    }();
    return runs;
}

void ov::npuw::perf::register_run_boundary_owner(const void* owner) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto& owners = boundary_owners();
    if (std::find(owners.begin(), owners.end(), owner) != owners.end()) {
        return;
    }
    owners.push_back(owner);
    static bool warned = false;
    if (owners.size() > 1u && !warned) {
        warned = true;
        LOG_WARN("NPUW profiling: more than one pipeline is being profiled; "
                 "per-run statistics will be mixed");
    }
}

void ov::npuw::perf::unregister_run_boundary_owner(const void* owner) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto& owners = boundary_owners();
    owners.erase(std::remove(owners.begin(), owners.end(), owner), owners.end());
}
