// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Dynamic-load wrapper around ocloc's in-memory oclocInvoke.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace cldnn {
namespace ocloc {

// From ocloc_api.h.
using ocloc_invoke_fn = int (*)(unsigned int numArgs, const char* argv[],
                                const uint32_t numSources, const uint8_t** dataSources,
                                const uint64_t* lenSources, const char** nameSources,
                                const uint32_t numInputHeaders, const uint8_t** dataInputHeaders,
                                const uint64_t* lenInputHeaders, const char** nameInputHeaders,
                                uint32_t* numOutputs, uint8_t*** dataOutputs,
                                uint64_t** lenOutputs, char*** nameOutputs);
using ocloc_free_output_fn = int (*)(uint32_t* numOutputs, uint8_t*** dataOutputs,
                                     uint64_t** lenOutputs, char*** nameOutputs);

struct ocloc_api {
    ocloc_invoke_fn invoke = nullptr;
    ocloc_free_output_fn free_output = nullptr;
    bool ok() const { return invoke != nullptr && free_output != nullptr; }
};

inline const char* getenv_or_null(const char* name) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? v : nullptr;
}

// Search order: $OV_GPU_OCLOC_PATH, then the library name.
inline const ocloc_api& load_ocloc() {
    static ocloc_api api;
    static std::once_flag once;
    std::call_once(once, [] {
#ifdef _WIN32
        HMODULE handle = nullptr;
        if (const char* p = getenv_or_null("OV_GPU_OCLOC_PATH"))
            handle = LoadLibraryA(p);
        if (!handle)
            handle = LoadLibraryA("ocloc64.dll");
        if (handle) {
            api.invoke = reinterpret_cast<ocloc_invoke_fn>(GetProcAddress(handle, "oclocInvoke"));
            api.free_output = reinterpret_cast<ocloc_free_output_fn>(GetProcAddress(handle, "oclocFreeOutput"));
        }
#else
        void* handle = nullptr;
        if (const char* p = getenv_or_null("OV_GPU_OCLOC_PATH"))
            handle = dlopen(p, RTLD_LAZY | RTLD_LOCAL);
        if (!handle)
            handle = dlopen("libocloc.so", RTLD_LAZY | RTLD_LOCAL);
        if (handle) {
            api.invoke = reinterpret_cast<ocloc_invoke_fn>(dlsym(handle, "oclocInvoke"));
            api.free_output = reinterpret_cast<ocloc_free_output_fn>(dlsym(handle, "oclocFreeOutput"));
        }
#endif
    });
    return api;
}

// Returns the .bin output, or an empty vector on failure.
inline std::vector<uint8_t> ocloc_compile_ocl_c(const std::string& source,
                                                const std::string& options,
                                                const std::string& device_arg,
                                                std::string* log_out = nullptr) {
    const ocloc_api& api = load_ocloc();
    if (!api.ok()) {
        if (log_out)
            *log_out += "[GPU offline] ocloc library not loaded\n";
        return {};
    }

    // oclocInvoke is not reentrant.
    static std::mutex ocloc_call_mutex;
    // TEMP: OV_GPU_OCLOC_NO_LOCK=1 skips the lock.
    static const bool no_lock = getenv_or_null("OV_GPU_OCLOC_NO_LOCK") != nullptr;
    std::unique_lock<std::mutex> call_lock(ocloc_call_mutex, std::defer_lock);
    if (!no_lock)
        call_lock.lock();

    std::vector<const char*> argv;
    argv.push_back("ocloc");
    argv.push_back("compile");
    argv.push_back("-device");
    argv.push_back(device_arg.c_str());
    // No SPIR-V, so a wrong target fails to load instead of being re-JITed.
    argv.push_back("-exclude_ir");
    // TEMP: ocloc defaults DG2 (12.55-12.57) to bindless, but its runtime expects bindful and rejects the ISA.
    if (device_arg.rfind("12.55.", 0) == 0 || device_arg.rfind("12.56.", 0) == 0 || device_arg.rfind("12.57.", 0) == 0) {
        argv.push_back("-stateful_address_mode");
        argv.push_back("bindful");
    }
    if (!options.empty()) {
        argv.push_back("-options");
        argv.push_back(options.c_str());
    }
    argv.push_back("-file");
    argv.push_back("kernel.cl");

    // Include the terminator: ocloc reads the source as a C string.
    const uint8_t* src_data[] = { reinterpret_cast<const uint8_t*>(source.data()) };
    const uint64_t src_len[]  = { static_cast<uint64_t>(source.size() + 1) };
    const char*    src_name[] = { "kernel.cl" };

    uint32_t num_out = 0;
    uint8_t** out_data = nullptr;
    uint64_t* out_len = nullptr;
    char**    out_name = nullptr;

    int rc = api.invoke(static_cast<unsigned>(argv.size()), argv.data(),
                        1, src_data, src_len, src_name,
                        0, nullptr, nullptr, nullptr,
                        &num_out, &out_data, &out_len, &out_name);

    std::vector<uint8_t> result;
    if (log_out)
        *log_out += "[GPU offline] rc=" + std::to_string(rc) + " numOutputs=" + std::to_string(num_out) + "\n";

    // Scan outputs even on failure to capture stdout.log.
#if defined(__clang__)
#pragma clang unsafe_buffer_usage begin
#endif
    if (out_data && out_len && out_name) {
        for (uint32_t i = 0; i < num_out; ++i) {
            const std::string name = out_name[i] ? out_name[i] : "";
            if (!out_data[i] || out_len[i] == 0)
                continue;
            if (log_out && name == "stdout.log")
                log_out->append(reinterpret_cast<const char*>(out_data[i]), static_cast<size_t>(out_len[i]));
            if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".bin") == 0)
                result.assign(out_data[i], out_data[i] + out_len[i]);
        }
    }
#if defined(__clang__)
#pragma clang unsafe_buffer_usage end
#endif

    if (api.free_output)
        api.free_output(&num_out, &out_data, &out_len, &out_name);

    return result;
}

}  // namespace ocloc
}  // namespace cldnn
