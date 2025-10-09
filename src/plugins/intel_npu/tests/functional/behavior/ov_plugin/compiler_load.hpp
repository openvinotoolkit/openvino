// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "behavior/ov_plugin/life_time.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/common_util.hpp"

namespace {

#ifdef _WIN32
#    include <ShlObj.h>
#    include <ntdef.h>
#    include <windows.h>

typedef struct _LDR_DLL_LOADED_NOTIFICATION_DATA {
    ULONG Flags;                   // Reserved.
    PCUNICODE_STRING FullDllName;  // The full path name of the DLL module.
    PCUNICODE_STRING BaseDllName;  // The base file name of the DLL module.
    PVOID DllBase;                 // A pointer to the base address for the DLL in memory.
    ULONG SizeOfImage;             // The size of the DLL image, in bytes.
} LDR_DLL_LOADED_NOTIFICATION_DATA, *PLDR_DLL_LOADED_NOTIFICATION_DATA;

typedef struct _LDR_DLL_UNLOADED_NOTIFICATION_DATA {
    ULONG Flags;                   // Reserved.
    PCUNICODE_STRING FullDllName;  // The full path name of the DLL module.
    PCUNICODE_STRING BaseDllName;  // The base file name of the DLL module.
    PVOID DllBase;                 // A pointer to the base address for the DLL in memory.
    ULONG SizeOfImage;             // The size of the DLL image, in bytes.
} LDR_DLL_UNLOADED_NOTIFICATION_DATA, *PLDR_DLL_UNLOADED_NOTIFICATION_DATA;

typedef union _LDR_DLL_NOTIFICATION_DATA {
    LDR_DLL_LOADED_NOTIFICATION_DATA Loaded;
    LDR_DLL_UNLOADED_NOTIFICATION_DATA Unloaded;
} LDR_DLL_NOTIFICATION_DATA, *PLDR_DLL_NOTIFICATION_DATA;

using PCLDR_DLL_NOTIFICATION_DATA = CONST PLDR_DLL_NOTIFICATION_DATA;
using PLDR_DLL_NOTIFICATION_FUNCTION = VOID (*)(ULONG, PCLDR_DLL_NOTIFICATION_DATA, PVOID);
using LdrRegisterDllNotification_Fnc = NTSTATUS (*)(ULONG, PLDR_DLL_NOTIFICATION_FUNCTION, PVOID, PVOID*);
using LdrUnregisterDllNotification_Fnc = NTSTATUS (*)(PVOID Cookie);

CONST ULONG LDR_DLL_NOTIFICATION_REASON_LOADED = 1;
CONST ULONG LDR_DLL_NOTIFICATION_REASON_UNLOADED = 2;

void register_Dll_notification_with_callback(LdrRegisterDllNotification_Fnc LdrRegisterDllNotification,
                                             const std::function<void(void)>& cb,
                                             LdrUnregisterDllNotification_Fnc LdrUnregisterDllNotification) {
    PVOID Cookie;
    static bool cidLibraryWasLoaded;
    static bool cipLibraryWasLoaded;
    static bool cipLibraryWasUnloaded;
    cidLibraryWasLoaded = cipLibraryWasLoaded = cipLibraryWasUnloaded = false;
    LdrRegisterDllNotification(
        0,
        [](ULONG NotificationReason, PCLDR_DLL_NOTIFICATION_DATA NotificationData, PVOID /* unusedContext */) {
            if (NotificationReason == LDR_DLL_NOTIFICATION_REASON_LOADED) {
                std::wstring_view wstr(NotificationData->Loaded.FullDllName->Buffer);
                if (wstr.find(L"npu_driver_compiler") != std::wstring_view::npos) {
                    cidLibraryWasLoaded = true;
                } else if (wstr.find(L"npu_mlir_compiler") != std::wstring_view::npos) {
                    cipLibraryWasLoaded = true;
                }
            } else if (NotificationReason == LDR_DLL_NOTIFICATION_REASON_UNLOADED) {
                std::wstring_view wstr(NotificationData->Unloaded.FullDllName->Buffer);
                if (wstr.find(L"npu_mlir_compiler") != std::wstring_view::npos) {
                    cipLibraryWasUnloaded = true;
                }
            }
        },
        NULL,
        &Cookie);
    cb();
    LdrUnregisterDllNotification(Cookie);
}

bool was_cid_loaded() {
    static bool cidLibraryWasLoaded;
    return cidLibraryWasLoaded;
}

bool was_cip_loaded() {
    static bool cipLibraryWasLoaded;
    static bool cipLibraryWasUnloaded;
    return cipLibraryWasLoaded && cipLibraryWasUnloaded;
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
    return dlopen("libnpu_driver_compiler.so", RTLD_LAZY | RTLD_NOLOAD) != NULL;
}

bool is_cip_loaded() {
    return dlopen("libnpu_mlir_compiler.so", RTLD_LAZY | RTLD_NOLOAD) != NULL;
}

void invalidate_umd_caching() {}

#endif

}  // namespace

using PropertiesParams = std::tuple<std::string,  // Device name
                                    ov::AnyMap    // Config
                                    >;

namespace ov {
namespace test {
namespace behavior {

class OVPostponeCompilerLoadTestsNPU : public OVPluginTestBase, public testing::WithParamInterface<PropertiesParams> {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(target_device, properties) = this->GetParam();
        APIBaseTest::SetUp();
        function = ov::test::utils::make_conv_pool_relu();
#ifdef _WIN32
        HMODULE hNtdll;
        ASSERT_TRUE(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, L"ntdll.dll", &hNtdll));
        ASSERT_TRUE(hNtdll != NULL);
        LdrRegisterDllNotification =
            reinterpret_cast<LdrRegisterDllNotification_Fnc>(GetProcAddress(hNtdll, "LdrRegisterDllNotification"));
        LdrUnregisterDllNotification =
            reinterpret_cast<LdrUnregisterDllNotification_Fnc>(GetProcAddress(hNtdll, "LdrRegisterDllNotification"));
        ASSERT_TRUE(LdrRegisterDllNotification != NULL && LdrUnregisterDllNotification != NULL);
#endif
    }

#ifdef _WIN32
    LdrRegisterDllNotification_Fnc LdrRegisterDllNotification;
    LdrUnregisterDllNotification_Fnc LdrUnregisterDllNotification;
#endif
    ov::AnyMap properties;
    std::shared_ptr<ov::Model> function;
};

using PropertiesWithArgumentsParamsNPU =
    std::tuple</* target_device = */ std::string, /* property_name = */ std::string, /* arguments = */ AnyMap>;

class OVPostponeCompilerLoadWithArgumentsTestsNPU
    : public OVPluginTestBase,
      public testing::WithParamInterface<PropertiesWithArgumentsParamsNPU> {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(target_device, property_name, arguments) = this->GetParam();
        APIBaseTest::SetUp();
        function = ov::test::utils::make_conv_pool_relu();
#ifdef _WIN32
        HMODULE hNtdll;
        ASSERT_TRUE(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, L"ntdll.dll", &hNtdll));
        ASSERT_TRUE(hNtdll != NULL);
        LdrRegisterDllNotification =
            reinterpret_cast<LdrRegisterDllNotification_Fnc>(GetProcAddress(hNtdll, "LdrRegisterDllNotification"));
        LdrUnregisterDllNotification =
            reinterpret_cast<LdrUnregisterDllNotification_Fnc>(GetProcAddress(hNtdll, "LdrRegisterDllNotification"));
        ASSERT_TRUE(LdrRegisterDllNotification != NULL && LdrUnregisterDllNotification != NULL);
#endif
    }

#ifdef _WIN32
    LdrRegisterDllNotification_Fnc LdrRegisterDllNotification;
    LdrUnregisterDllNotification_Fnc LdrUnregisterDllNotification;
#endif
    std::string property_name;
    ov::AnyMap arguments;
    std::shared_ptr<ov::Model> function;
};

using OVSetRunTimePropertiesTestsNPU = OVPostponeCompilerLoadTestsNPU;
using OVSetCompileTimePropertiesTestsNPU = OVPostponeCompilerLoadTestsNPU;
using OVSetBothPropertiesTestsNPU = OVPostponeCompilerLoadTestsNPU;
using OVSetCartesianProductPropertiesTestsNPU = OVPostponeCompilerLoadTestsNPU;
using OVRunTimePropertiesArgumentsTestsNPU = OVPostponeCompilerLoadWithArgumentsTestsNPU;
using OVCompileTimePropertiesArgumentsTestsNPU = OVPostponeCompilerLoadWithArgumentsTestsNPU;
using OVBothPropertiesArgumentsTestsNPU = OVPostponeCompilerLoadWithArgumentsTestsNPU;

TEST_P(OVRunTimePropertiesArgumentsTestsNPU, GetRunTimePropertiesNoCompilerLoad) {
    invalidate_umd_caching();
    ov::Core core = ov::test::utils::create_core();
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(core.get_property(target_device, property_name, arguments));
        },
        this->LdrUnregisterDllNotification);
    ASSERT_TRUE(!was_cid_loaded() && !was_cip_loaded());
}

TEST_P(OVSetRunTimePropertiesTestsNPU, SetRunTimePropertiesNoCompilerLoad) {
    invalidate_umd_caching();
    ov::Core core = ov::test::utils::create_core();
    core.get_property(target_device, ov::available_devices);
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(core.set_property(target_device, properties));
        },
        this->LdrUnregisterDllNotification);
    ASSERT_TRUE(!was_cid_loaded() && !was_cip_loaded());
}

TEST_P(OVCompileTimePropertiesArgumentsTestsNPU, GetCompileTimePropertiesCompilerLoad) {
    invalidate_umd_caching();
    ov::Core core = ov::test::utils::create_core();
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(core.get_property(target_device, property_name, arguments));
        },
        this->LdrUnregisterDllNotification);
    if (arguments.find(ov::intel_npu::compiler_type.name()) != arguments.end()) {
        ASSERT_TRUE(!was_cid_loaded() && was_cip_loaded());
    } else {
        ASSERT_TRUE(was_cid_loaded() && !was_cip_loaded());
    }
}

TEST_P(OVSetCompileTimePropertiesTestsNPU, SetCompileTimePropertiesCompilerLoad) {
    invalidate_umd_caching();
    ov::Core core = ov::test::utils::create_core();
    core.get_property(target_device, ov::available_devices);
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(core.set_property(target_device, properties));
        },
        this->LdrUnregisterDllNotification);
    ASSERT_TRUE(was_cid_loaded() && !was_cip_loaded());
}

TEST_P(OVBothPropertiesArgumentsTestsNPU, GetBothPropertiesCompilerLoad) {
    invalidate_umd_caching();
    ov::Core core = ov::test::utils::create_core();
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(core.get_property(target_device, property_name, arguments));
        },
        this->LdrUnregisterDllNotification);
    if (property_name == ov::log::level.name()) {  // special case
        ASSERT_TRUE(!was_cid_loaded() && !was_cip_loaded());
    } else if (arguments.find(ov::intel_npu::compiler_type.name()) != arguments.end()) {
        ASSERT_TRUE(!was_cid_loaded() && was_cip_loaded());
    } else {
        ASSERT_TRUE(was_cid_loaded() && !was_cip_loaded());
    }
}

TEST_P(OVSetBothPropertiesTestsNPU, SetBothPropertiesCompilerLoad) {
    invalidate_umd_caching();
    ov::Core core = ov::test::utils::create_core();
    core.get_property(target_device, ov::available_devices);
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(core.set_property(target_device, properties));
        },
        this->LdrUnregisterDllNotification);
    if (properties.find(ov::log::level.name()) != properties.end()) {  // special case
        ASSERT_TRUE(!was_cid_loaded() && !was_cip_loaded());
    } else {
        ASSERT_TRUE(was_cid_loaded() && !was_cip_loaded());
    }
}

TEST_P(OVSetCartesianProductPropertiesTestsNPU, SetCartesianProductPropertiesCompilerLoad) {
    invalidate_umd_caching();
    ov::Core core = ov::test::utils::create_core();
    core.get_property(target_device, ov::available_devices);
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(core.set_property(target_device, properties));
        },
        this->LdrUnregisterDllNotification);
    if (properties.find(ov::intel_npu::compiler_type.name()) != properties.end()) {
        ASSERT_TRUE(!was_cid_loaded() && was_cip_loaded());
    } else {
        ASSERT_TRUE(was_cid_loaded() && !was_cip_loaded());
    }
}

TEST_P(OVPostponeCompilerLoadTestsNPU, CompileModelDoesCompilerLoad) {
    invalidate_umd_caching();
    ov::Core core = ov::test::utils::create_core();
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(core.compile_model(function, target_device, properties));
        },
        this->LdrUnregisterDllNotification);
    if (properties.find(ov::intel_npu::compiler_type.name()) != properties.end()) {
        ASSERT_TRUE(!was_cid_loaded() && was_cip_loaded());
    } else {
        ASSERT_TRUE(was_cid_loaded() && !was_cip_loaded());
    }
}

TEST_P(OVPostponeCompilerLoadTestsNPU, CompileModelWithCacheDoesNotCompilerLoad) {
    invalidate_umd_caching();
    ov::Core core = ov::test::utils::create_core();
    const std::string cache_dir_name = "cache_dir";
    properties[ov::cache_dir.name()] = cache_dir_name;
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(core.compile_model(function, target_device, properties));
        },
        this->LdrUnregisterDllNotification);
    if (properties.find(ov::intel_npu::compiler_type.name()) != properties.end()) {
        ASSERT_TRUE(!was_cid_loaded() && was_cip_loaded());
    } else {
        ASSERT_TRUE(was_cid_loaded() && !was_cip_loaded());
    }

    core = ov::test::utils::create_core();
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(core.compile_model(function, target_device, properties));
        },
        this->LdrUnregisterDllNotification);
    // ASSERT_TRUE(!was_cid_loaded() && !was_cip_loaded());  // further optimization in other PR
    if (properties.find(ov::intel_npu::compiler_type.name()) != properties.end()) {
        ASSERT_TRUE(!was_cid_loaded() && was_cip_loaded());
    } else {
        ASSERT_TRUE(was_cid_loaded() && !was_cip_loaded());
    }

    ov::util::iterate_files(
        cache_dir_name,
        [](const std::string& file, bool is_dir) {
            if (is_dir) {
                ov::test::utils::removeDir(file);
            } else {
                ov::test::utils::removeFile(file);
            }
        },
        /* recurse = */ true);
}

TEST_P(OVPostponeCompilerLoadTestsNPU, ImportModelDoesNotCompilerLoad) {
    invalidate_umd_caching();
    ov::Core core = ov::test::utils::create_core();
    std::stringstream ss;
    ov::CompiledModel compiled_model;
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(compiled_model = core.compile_model(function, target_device, properties));
        },
        this->LdrUnregisterDllNotification);
    if (properties.find(ov::intel_npu::compiler_type.name()) != properties.end()) {
        ASSERT_TRUE(!was_cid_loaded() && was_cip_loaded());
    } else {
        ASSERT_TRUE(was_cid_loaded() && !was_cip_loaded());
    }
    compiled_model.export_model(ss);

    compiled_model = core.import_model(ss, target_device, properties);
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(compiled_model = core.import_model(ss, target_device, properties));
        },
        this->LdrUnregisterDllNotification);
    // ASSERT_TRUE(!is_cid_loaded() && !is_cip_loaded());  // further optimization in other PR
    if (properties.find(ov::intel_npu::compiler_type.name()) != properties.end()) {
        ASSERT_TRUE(!was_cid_loaded() && was_cip_loaded());
    } else {
        ASSERT_TRUE(was_cid_loaded() && !was_cip_loaded());
    }
}

TEST_P(OVPostponeCompilerLoadTestsNPU, QueryModelDoesCompilerLoad) {
    invalidate_umd_caching();
    ov::Core core = ov::test::utils::create_core();
    register_Dll_notification_with_callback(
        this->LdrRegisterDllNotification,
        [&, this]() {
            OV_ASSERT_NO_THROW(core.query_model(function, target_device, properties));
        },
        this->LdrUnregisterDllNotification);
    if (properties.find(ov::intel_npu::compiler_type.name()) != properties.end()) {
        ASSERT_TRUE(!was_cid_loaded() && was_cip_loaded());
    } else {
        ASSERT_TRUE(was_cid_loaded() && !was_cip_loaded());
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
