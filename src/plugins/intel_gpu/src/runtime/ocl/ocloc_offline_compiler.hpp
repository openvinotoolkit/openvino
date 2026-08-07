// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// POC ONLY — dynamic-load wrapper around NEO ocloc's in-memory `oclocInvoke`.
//
// Purpose: validate that the OV GPU plugin can compile a kernel's OpenCL C source
// through NEO ocloc's offline (HW-free) path (oclocInvoke -> IGC), instead of the
// driver JIT (clBuildProgram). This wrapper only compiles source -> native binary
// bytes; it does NOT create cl::Kernel objects. See the POC plan.
//
// No build-system change: ocloc is loaded at runtime via LoadLibrary/dlopen and the
// oclocInvoke/oclocFreeOutput signatures are declared locally (copied from
// compute-runtime/shared/offline_compiler/source/ocloc_api.h), so we do NOT include
// any compute-runtime header.

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
namespace ocloc_poc {

// Signatures copied verbatim from ocloc_api.h (extern "C").
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

// Loads ocloc once (thread-safe) and caches the resolved function pointers.
// Search order for the shared library:
//   1) $OV_GPU_OCLOC_PATH (full path to the DLL/SO), if set
//   2) plain name via system loader ("ocloc64.dll" / "libocloc.so")
//   3) known Intel oneAPI install path (Windows)
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
        if (!handle)
            handle = LoadLibraryA("C:\\Program Files (x86)\\Intel\\oneAPI\\ocloc\\2025.2\\bin\\ocloc64.dll");
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

// Compiles a single OpenCL C source string to a device native binary via ocloc.
// `device_arg` is passed to ocloc `-device` (e.g. "0x4680" or "12.2.0").
// `options` (OpenCL build flags) are forwarded via ocloc `-options`.
// Returns the ocloc `.bin` output on success, or empty vector on any failure.
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

    // oclocInvoke drives IGC/FCL global/process-wide state and is not reentrant.
    // build_batch runs batches in parallel (kernels_cache task executor), so
    // serialize all ocloc calls to avoid intermittent failures / empty output.
    static std::mutex ocloc_call_mutex;
    std::lock_guard<std::mutex> call_lock(ocloc_call_mutex);

    // argv: ocloc compile -device <id> [-options "<opts>"] -file kernel.cl
    std::vector<const char*> argv;
    argv.push_back("ocloc");
    argv.push_back("compile");
    argv.push_back("-device");
    argv.push_back(device_arg.c_str());
    // -exclude_ir: emit native-ISA-only zebin (no embedded SPIR-V). Without this, ocloc emits
    // a "fat" zebin (native ISA + SPIR-V IR); loading it on a mismatched device makes the driver
    // silently re-JIT from the IR (masking a target mismatch). With IR excluded, a wrong-target
    // binary fails to load -- which is what we want to validate that the native binary is used.
    argv.push_back("-exclude_ir");
    if (!options.empty()) {
        argv.push_back("-options");
        argv.push_back(options.c_str());
    }
    argv.push_back("-file");
    argv.push_back("kernel.cl");

    // NOTE: ocloc's in-memory source path (OclocArgHelper::loadDataFromFile) copies
    // exactly `length` bytes and does NOT null-terminate, yet offline_compiler.cpp
    // reads the buffer as a C-string (strstr / char*->std::string) -> heap over-read
    // -> intermittent garbage ("source file is not valid UTF-8" past real EOF).
    // Work around it by including the string's guaranteed NUL terminator in `length`
    // (std::string::data()[size()] == '\0', reading it is well-defined).
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

    // Scan outputs even when rc != 0 so a valid .bin is not discarded, and so we
    // can capture ocloc's stdout.log (the failure reason). ocloc returns raw C
    // arrays; index them behind the clang unsafe-buffer pragma (POC-local).
#if defined(__clang__)
#pragma clang unsafe_buffer_usage begin
#endif
    if (out_data && out_len && out_name) {
        int best = -1;
        bool found_bin = false;
        for (uint32_t i = 0; i < num_out; ++i) {
            std::string name = out_name[i] ? out_name[i] : "";
            uint64_t len = out_len[i];
            if (log_out)
                *log_out += "  [" + std::to_string(i) + "] " + name + " (" + std::to_string(len) + " bytes)\n";
            if (log_out && name == "stdout.log" && out_data[i] && len > 0) {
                *log_out += "--- ocloc stdout.log ---\n";
                log_out->append(reinterpret_cast<const char*>(out_data[i]), static_cast<size_t>(len));
                if (!log_out->empty() && log_out->back() != '\n')
                    *log_out += "\n";
            }
            bool is_bin = name.size() >= 4 && name.compare(name.size() - 4, 4, ".bin") == 0;
            if (is_bin) {
                best = static_cast<int>(i);
                found_bin = true;
            } else if (!found_bin && name != "stdout.log" && (best < 0 || len > out_len[best])) {
                best = static_cast<int>(i);
            }
        }
        if (best >= 0 && out_data[best] && out_len[best] > 0)
            result.assign(out_data[best], out_data[best] + out_len[best]);
    }
#if defined(__clang__)
#pragma clang unsafe_buffer_usage end
#endif

    if (api.free_output)
        api.free_output(&num_out, &out_data, &out_len, &out_name);

    return result;
}

}  // namespace ocloc_poc
}  // namespace cldnn