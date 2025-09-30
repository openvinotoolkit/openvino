//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "overload/ov_plugin/properties_tests.hpp"

#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace {
#ifdef _WIN32
#    include <ShlObj.h>
#    include <windows.h>
bool is_cid_loaded() {
    HMODULE phModule = NULL;
    size_t ret =
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, L"npu_driver_compiler.dll", &phModule);
    return ret && phModule != NULL;
}

bool is_cip_loaded() {
    HMODULE phModule = NULL;
    size_t ret = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, L"npu_mlir_compiler.dll", &phModule);
    return ret && phModule != NULL;
}

void invalidate_umd_caching() {
    HMODULE _shell32 = LoadLibraryExW(L"shell32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    HRESULT(WINAPI * SHGetKnownFolderPath)
    (REFKNOWNFOLDERID, DWORD, HANDLE, PWSTR*) =
        _shell32 != nullptr
            ? reinterpret_cast<decltype(SHGetKnownFolderPath)>(GetProcAddress(_shell32, "SHGetKnownFolderPath"))
            : nullptr;
    if (SHGetKnownFolderPath != nullptr) {
        wchar_t* local = nullptr;
        if (SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, NULL, &local) == S_OK) {
            auto path = std::filesystem::path(L"\\\\?\\" + std::wstring(local) + L"\\Intel\\NPU");
            ov::util::iterate_files(
                path.string(),
                [](const std::string& file, bool is_dir) {
                    if (is_dir) {
                        ov::test::utils::removeDir(file);
                    } else {
                        ov::test::utils::removeFile(file);
                    }
                },
                /* recurse = */ true);
            CoTaskMemFree(local);
        }
        FreeLibrary(_shell32);
    }
}

#elif defined(__linux__)
#    include <dlfcn.h>
bool is_cid_loaded() {
    return dlopen("libnpu_driver_compiler.so", RTLD_NOLOAD) != NULL;
}

bool is_cip_loaded() {
    return dlopen("libnpu_mlir_compiler.so", RTLD_NOLOAD) != NULL;
}

void invalidate_umd_caching() {}

#endif

}  // namespace

namespace ov::test::behavior {

TEST_P(OVPropertiesTestsNPU, GetSupportedPropertiesImplyCompilerLoad) {
    invalidate_umd_caching();
    core->get_property(target_device, ov::supported_properties, properties);
    if (auto it = properties.find(ov::intel_npu::compiler_type.name());
        it != properties.end() && it->second == ov::intel_npu::CompilerType::MLIR) {
        ASSERT_TRUE(is_cip_loaded());
    } else {
        ASSERT_TRUE(is_cid_loaded());
    }
}

const std::vector<ov::AnyMap> runtimeProperties = {{{ov::cache_dir.name(), ""}}};

const std::vector<ov::AnyMap> compileTimeProperties = {
    {{ov::intel_npu::dma_engines.name(), 1}},
};

const std::vector<ov::AnyMap> bothProperties = {
    {{ov::log::level.name(), ov::log::Level::INFO}},
};

}  // namespace ov::test::behavior

namespace {
template <typename T>
constexpr std::vector<T> operator+(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    std::vector<T> result;
    result.insert(result.end(), vec1.begin(), vec1.end());
    result.insert(result.end(), vec2.begin(), vec2.end());
    return result;
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(runtimeProperties + compileTimeProperties +
                                                                bothProperties)),
                         (ov::test::utils::appendPlatformTypeTestName<OVPropertiesTestsNPU>));
}  // namespace
