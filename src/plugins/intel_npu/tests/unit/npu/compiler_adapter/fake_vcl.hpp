// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "intel_npu/utils/vcl/vcl_api.hpp"

namespace fake_vcl {

//
// The VCL entry points are plain C function pointers with no user-data parameter, so the fake has to
// route through a file-static "current instance" pointer. FakeVcl sets it on construction and clears
// it on destruction, which keeps one fake per test.
//
class FakeVcl;
inline FakeVcl*& current() {
    static FakeVcl* instance = nullptr;
    return instance;
}

/// State objects the opaque VCL handles point at. Never dereferenced by production code, so empty
/// structs with distinct addresses are enough.
struct FakeCompilerState {
    int tag = 1;
};
struct FakeExecutableState {
    int tag = 2;
};
struct FakeQueryState {
    int tag = 3;
};
struct FakeProfilingState {
    int tag = 4;
};
struct FakeLogState {
    int tag = 5;
};

class FakeVcl {
public:
    FakeVcl() {
        current() = this;
        _api = std::make_shared<intel_npu::VCLApi>(intel_npu::VCLApi::NoLoad{});
        wire();
    }

    ~FakeVcl() {
        current() = nullptr;
    }

    FakeVcl(const FakeVcl&) = delete;
    FakeVcl& operator=(const FakeVcl&) = delete;

    /// The dispatch table to inject into VCLCompilerImpl.
    std::shared_ptr<const intel_npu::VCLApi> api() const {
        return _api;
    }
    std::shared_ptr<intel_npu::VCLApi> mutableApi() const {
        return _api;
    }

    //
    // --- scripting ---
    //

    /// Force a specific entry point to fail. Cleared per entry point.
    void failWith(const std::string& fn, vcl_result_t result) {
        _results[fn] = result;
    }
    void succeed(const std::string& fn) {
        _results.erase(fn);
    }

    vcl_result_t resultFor(const std::string& fn) {
        calls.push_back(fn);
        const auto it = _results.find(fn);
        return it == _results.end() ? VCL_RESULT_SUCCESS : it->second;
    }

    //
    // --- inputs the fake reports back to production code ---
    //

    vcl_version_info_t reportedCompilerVersion{VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR};
    vcl_version_info_t reportedProfilingVersion{VCL_PROFILING_VERSION_MAJOR, VCL_PROFILING_VERSION_MINOR};
    vcl_version_info_t propertiesVersion{VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR};
    uint32_t supportedOpsets = 11;
    std::string compilerId = "fake-vcl";

    std::string logString = "fake vcl log";

    /// Blob bytes handed back by vclAllocatedExecutableCreate4.
    std::vector<uint8_t> blobPayload{1, 2, 3, 4, 5};
    /// Per-blob sizes handed back by vclAllocatedExecutableCreateWSOneShot2 (init schedules, then main).
    std::vector<size_t> wsBlobSizes{8, 16, 32};

    /// nullopt makes vclExecutableGetCompatibilityString report UNSUPPORTED_FEATURE.
    std::optional<std::string> compatibilityString = std::string("fake-compat");
    /// When set, the reported size is used instead of compatibilityString->size() + 1.
    std::optional<uint64_t> compatibilityStringSizeOverride;

    /// Raw buffer returned by vclGetCompilerSupportedOptions (trailing NULs are meaningful).
    std::string supportedOptionsBuffer = std::string("OPT_A OPT_B OPT_C");

    /// Raw buffer returned by vclQueryNetwork.
    std::vector<char> queryResultBuffer;

    /// Bytes returned by vclGetDecodedProfilingBuffer; nullptr data if empty and forceNullProfData.
    std::vector<uint8_t> profilingPayload;
    bool forceNullProfilingData = false;

    //
    // --- recordings ---
    //

    std::vector<std::string> calls;
    std::vector<vcl_device_desc_t> deviceDescs;
    std::vector<vcl_compiler_desc_t> compilerDescs;
    /// The build-flags string handed to VCL, per executable-creating call.
    std::vector<std::string> buildFlags;
    /// Model IR bytes handed to VCL, per executable-creating call.
    std::vector<uint64_t> modelIrSizes;
    std::vector<std::string> queryBuildFlags;
    /// (option, value) pairs seen by vclGetCompilerIsOptionSupported; value is nullopt for nullptr.
    std::vector<std::pair<std::string, std::optional<std::string>>> optionSupportQueries;

    int compilerDestroyCount = 0;
    int executableDestroyCount = 0;
    int queryDestroyCount = 0;
    int profilingDestroyCount = 0;

    size_t callCount(const std::string& fn) const {
        size_t n = 0;
        for (const auto& c : calls) {
            if (c == fn) {
                ++n;
            }
        }
        return n;
    }

    bool called(const std::string& fn) const {
        return callCount(fn) > 0;
    }

    /// Index of the first occurrence of fn in the call log, or npos.
    size_t indexOf(const std::string& fn) const {
        for (size_t i = 0; i < calls.size(); ++i) {
            if (calls[i] == fn) {
                return i;
            }
        }
        return std::string::npos;
    }

    // Handle-backing state, kept alive for the fake's lifetime.
    FakeCompilerState compilerState;
    FakeExecutableState executableState;
    FakeQueryState queryState;
    FakeProfilingState profilingState;
    FakeLogState logState;

private:
    void wire();

    std::shared_ptr<intel_npu::VCLApi> _api;
    std::map<std::string, vcl_result_t> _results;
};

//
// --- thunks ---
//

#define FAKE_GUARD(fn)                                  \
    FakeVcl* self = current();                          \
    if (self == nullptr) {                              \
        return VCL_RESULT_ERROR_UNKNOWN;                \
    }                                                   \
    const vcl_result_t scripted = self->resultFor(#fn); \
    if (scripted != VCL_RESULT_SUCCESS) {               \
        return scripted;                                \
    }

inline vcl_result_t VCL_APICALL fake_vclGetVersion(vcl_version_info_t* compilerVersion,
                                                   vcl_version_info_t* profilingVersion) {
    FAKE_GUARD(vclGetVersion)
    if (compilerVersion != nullptr) {
        *compilerVersion = self->reportedCompilerVersion;
    }
    if (profilingVersion != nullptr) {
        *profilingVersion = self->reportedProfilingVersion;
    }
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclCompilerCreate(vcl_compiler_desc_t* compilerDesc,
                                                       vcl_device_desc_t* deviceDesc,
                                                       vcl_compiler_handle_t* compiler,
                                                       vcl_log_handle_t* logHandle) {
    FakeVcl* self = current();
    if (self == nullptr) {
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    // Record the descriptors even on the failure path: the test asserts what was forwarded.
    if (compilerDesc != nullptr) {
        self->compilerDescs.push_back(*compilerDesc);
    }
    if (deviceDesc != nullptr) {
        self->deviceDescs.push_back(*deviceDesc);
    }
    const vcl_result_t scripted = self->resultFor("vclCompilerCreate");
    if (scripted != VCL_RESULT_SUCCESS) {
        return scripted;
    }
    if (compiler != nullptr) {
        *compiler = reinterpret_cast<vcl_compiler_handle_t>(&self->compilerState);
    }
    if (logHandle != nullptr) {
        *logHandle = reinterpret_cast<vcl_log_handle_t>(&self->logState);
    }
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclCompilerDestroy(vcl_compiler_handle_t) {
    FakeVcl* self = current();
    if (self == nullptr) {
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    ++self->compilerDestroyCount;
    return self->resultFor("vclCompilerDestroy");
}

inline vcl_result_t VCL_APICALL fake_vclCompilerGetProperties(vcl_compiler_handle_t,
                                                              vcl_compiler_properties_t* properties) {
    FAKE_GUARD(vclCompilerGetProperties)
    if (properties != nullptr) {
        properties->id = self->compilerId.c_str();
        properties->version = self->propertiesVersion;
        properties->supportedOpsets = self->supportedOpsets;
    }
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclLogHandleGetString(vcl_log_handle_t, size_t* logSize, char* log) {
    FAKE_GUARD(vclLogHandleGetString)
    if (logSize == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (log == nullptr) {
        *logSize = self->logString.size();
        return VCL_RESULT_SUCCESS;
    }
    std::memcpy(log, self->logString.data(), self->logString.size());
    *logSize = self->logString.size();
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclAllocatedExecutableCreate4(vcl_compiler_handle_t,
                                                                   vcl_executable_desc_t desc,
                                                                   vcl_allocator2_t* allocator,
                                                                   uint8_t** blobBuffer,
                                                                   uint64_t* blobSize,
                                                                   vcl_executable_handle_t* executable) {
    FakeVcl* self = current();
    if (self == nullptr) {
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    if (desc.options != nullptr) {
        self->buildFlags.emplace_back(desc.options, static_cast<size_t>(desc.optionsSize));
    }
    self->modelIrSizes.push_back(desc.modelIRSize);

    const vcl_result_t scripted = self->resultFor("vclAllocatedExecutableCreate4");
    if (scripted != VCL_RESULT_SUCCESS) {
        return scripted;
    }
    if (allocator == nullptr || blobBuffer == nullptr || blobSize == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    const size_t size = self->blobPayload.size();
    uint8_t* buffer = allocator->allocate(allocator, size);
    if (buffer == nullptr) {
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }
    std::memcpy(buffer, self->blobPayload.data(), size);
    *blobBuffer = buffer;
    *blobSize = size;
    if (executable != nullptr) {
        *executable = reinterpret_cast<vcl_executable_handle_t>(&self->executableState);
    }
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclAllocatedExecutableCreateWSOneShot2(vcl_compiler_handle_t,
                                                                            vcl_executable_desc_t desc,
                                                                            vcl_allocator2_t* allocator,
                                                                            vcl_executable_handle_t* executable) {
    FakeVcl* self = current();
    if (self == nullptr) {
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    if (desc.options != nullptr) {
        self->buildFlags.emplace_back(desc.options, static_cast<size_t>(desc.optionsSize));
    }
    self->modelIrSizes.push_back(desc.modelIRSize);

    const vcl_result_t scripted = self->resultFor("vclAllocatedExecutableCreateWSOneShot2");
    if (scripted != VCL_RESULT_SUCCESS) {
        return scripted;
    }
    if (allocator == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    // One allocation per blob: init schedules first, main last.
    for (size_t size : self->wsBlobSizes) {
        uint8_t* buffer = allocator->allocate(allocator, size);
        if (buffer == nullptr) {
            return VCL_RESULT_ERROR_OUT_OF_MEMORY;
        }
        std::memset(buffer, static_cast<int>(size & 0xFF), size);
    }
    if (executable != nullptr) {
        *executable = reinterpret_cast<vcl_executable_handle_t>(&self->executableState);
    }
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclExecutableGetCompatibilityString(vcl_executable_handle_t,
                                                                         char* buffer,
                                                                         uint64_t* size) {
    FakeVcl* self = current();
    if (self == nullptr) {
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    const vcl_result_t scripted = self->resultFor("vclExecutableGetCompatibilityString");
    if (scripted != VCL_RESULT_SUCCESS) {
        return scripted;
    }
    if (!self->compatibilityString.has_value()) {
        return VCL_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
    if (size == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    // VCL reports the size including the NUL terminator.
    const std::string& value = *self->compatibilityString;
    const uint64_t reported = self->compatibilityStringSizeOverride.value_or(static_cast<uint64_t>(value.size()) + 1u);
    if (buffer == nullptr) {
        *size = reported;
        return VCL_RESULT_SUCCESS;
    }
    const size_t toCopy = static_cast<size_t>(reported == 0 ? 0 : reported - 1);
    std::memcpy(buffer, value.data(), std::min(toCopy, value.size()));
    if (reported > 0) {
        buffer[reported - 1] = '\0';
    }
    *size = reported;
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclExecutableDestroy(vcl_executable_handle_t) {
    FakeVcl* self = current();
    if (self == nullptr) {
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    ++self->executableDestroyCount;
    return self->resultFor("vclExecutableDestroy");
}

inline vcl_result_t VCL_APICALL fake_vclQueryNetworkCreate(vcl_compiler_handle_t,
                                                           vcl_query_desc_t desc,
                                                           vcl_query_handle_t* query) {
    FakeVcl* self = current();
    if (self == nullptr) {
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    if (desc.options != nullptr) {
        self->queryBuildFlags.emplace_back(desc.options, static_cast<size_t>(desc.optionsSize));
    }
    const vcl_result_t scripted = self->resultFor("vclQueryNetworkCreate");
    if (scripted != VCL_RESULT_SUCCESS) {
        return scripted;
    }
    if (query != nullptr) {
        *query = reinterpret_cast<vcl_query_handle_t>(&self->queryState);
    }
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclQueryNetwork(vcl_query_handle_t, uint8_t* queryResult, uint64_t* size) {
    FAKE_GUARD(vclQueryNetwork)
    if (size == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (queryResult == nullptr) {
        *size = self->queryResultBuffer.size();
        return VCL_RESULT_SUCCESS;
    }
    std::memcpy(queryResult, self->queryResultBuffer.data(), self->queryResultBuffer.size());
    *size = self->queryResultBuffer.size();
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclQueryNetworkDestroy(vcl_query_handle_t) {
    FakeVcl* self = current();
    if (self == nullptr) {
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    ++self->queryDestroyCount;
    return self->resultFor("vclQueryNetworkDestroy");
}

inline vcl_result_t VCL_APICALL fake_vclProfilingCreate(p_vcl_profiling_input_t,
                                                        vcl_profiling_handle_t* profilingHandle,
                                                        vcl_log_handle_t* logHandle) {
    FAKE_GUARD(vclProfilingCreate)
    if (profilingHandle != nullptr) {
        *profilingHandle = reinterpret_cast<vcl_profiling_handle_t>(&self->profilingState);
    }
    if (logHandle != nullptr) {
        *logHandle = reinterpret_cast<vcl_log_handle_t>(&self->logState);
    }
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclProfilingGetProperties(vcl_profiling_handle_t,
                                                               vcl_profiling_properties_t* properties) {
    FAKE_GUARD(vclProfilingGetProperties)
    if (properties != nullptr) {
        properties->version = self->reportedProfilingVersion;
    }
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclGetDecodedProfilingBuffer(vcl_profiling_handle_t,
                                                                  vcl_profiling_request_type_t,
                                                                  p_vcl_profiling_output_t output) {
    FAKE_GUARD(vclGetDecodedProfilingBuffer)
    if (output == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (self->forceNullProfilingData) {
        output->data = nullptr;
        output->size = 0;
        return VCL_RESULT_SUCCESS;
    }
    output->data = self->profilingPayload.data();
    output->size = self->profilingPayload.size();
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclProfilingDestroy(vcl_profiling_handle_t) {
    FakeVcl* self = current();
    if (self == nullptr) {
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    ++self->profilingDestroyCount;
    return self->resultFor("vclProfilingDestroy");
}

inline vcl_result_t VCL_APICALL fake_vclGetCompilerSupportedOptions(vcl_compiler_handle_t,
                                                                    char* result,
                                                                    uint64_t* size) {
    FAKE_GUARD(vclGetCompilerSupportedOptions)
    if (size == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    const std::string& buffer = self->supportedOptionsBuffer;
    if (result == nullptr) {
        *size = buffer.size();
        return VCL_RESULT_SUCCESS;
    }
    std::memcpy(result, buffer.data(), buffer.size());
    *size = buffer.size();
    return VCL_RESULT_SUCCESS;
}

inline vcl_result_t VCL_APICALL fake_vclGetCompilerIsOptionSupported(vcl_compiler_handle_t,
                                                                     const char* option,
                                                                     const char* value) {
    FakeVcl* self = current();
    if (self == nullptr) {
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    self->optionSupportQueries.emplace_back(
        option != nullptr ? std::string(option) : std::string(),
        value != nullptr ? std::optional<std::string>(std::string(value)) : std::nullopt);
    return self->resultFor("vclGetCompilerIsOptionSupported");
}

// Entry points the production code under test never calls; wired so a stray call is obvious.
inline vcl_result_t VCL_APICALL fake_vclExecutableCreate(vcl_compiler_handle_t,
                                                         vcl_executable_desc_t,
                                                         vcl_executable_handle_t*) {
    FakeVcl* self = current();
    return self == nullptr ? VCL_RESULT_ERROR_UNKNOWN : self->resultFor("vclExecutableCreate");
}

inline vcl_result_t VCL_APICALL fake_vclExecutableGetSerializableBlob(vcl_executable_handle_t,
                                                                      uint8_t*,
                                                                      uint64_t* blobSize) {
    FakeVcl* self = current();
    if (self == nullptr) {
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    if (blobSize != nullptr) {
        *blobSize = 0;
    }
    return self->resultFor("vclExecutableGetSerializableBlob");
}

#undef FAKE_GUARD

inline void FakeVcl::wire() {
    _api->vclGetVersion = &fake_vclGetVersion;
    _api->vclCompilerCreate = &fake_vclCompilerCreate;
    _api->vclCompilerDestroy = &fake_vclCompilerDestroy;
    _api->vclCompilerGetProperties = &fake_vclCompilerGetProperties;
    _api->vclQueryNetworkCreate = &fake_vclQueryNetworkCreate;
    _api->vclQueryNetwork = &fake_vclQueryNetwork;
    _api->vclQueryNetworkDestroy = &fake_vclQueryNetworkDestroy;
    _api->vclExecutableCreate = &fake_vclExecutableCreate;
    _api->vclExecutableDestroy = &fake_vclExecutableDestroy;
    _api->vclExecutableGetSerializableBlob = &fake_vclExecutableGetSerializableBlob;
    _api->vclProfilingCreate = &fake_vclProfilingCreate;
    _api->vclGetDecodedProfilingBuffer = &fake_vclGetDecodedProfilingBuffer;
    _api->vclProfilingDestroy = &fake_vclProfilingDestroy;
    _api->vclProfilingGetProperties = &fake_vclProfilingGetProperties;
    _api->vclLogHandleGetString = &fake_vclLogHandleGetString;
    _api->vclAllocatedExecutableCreate4 = &fake_vclAllocatedExecutableCreate4;
    _api->vclExecutableGetCompatibilityString = &fake_vclExecutableGetCompatibilityString;
    _api->vclGetCompilerSupportedOptions = &fake_vclGetCompilerSupportedOptions;
    _api->vclGetCompilerIsOptionSupported = &fake_vclGetCompilerIsOptionSupported;
    _api->vclAllocatedExecutableCreateWSOneShot2 = &fake_vclAllocatedExecutableCreateWSOneShot2;
    // Weak symbols the production path does not use stay null, matching an older library.
    _api->vclAllocatedExecutableCreate = nullptr;
    _api->vclAllocatedExecutableCreate2 = nullptr;
}

}  // namespace fake_vcl
