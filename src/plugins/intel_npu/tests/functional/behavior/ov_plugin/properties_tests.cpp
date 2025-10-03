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
    return dlopen("libnpu_driver_compiler.so", RTLD_LAZY | RTLD_NOLOAD) != NULL;
}

bool is_cip_loaded() {
    return dlopen("libnpu_mlir_compiler.so", RTLD_LAZY | RTLD_NOLOAD) != NULL;
}

void invalidate_umd_caching() {}

#endif

}  // namespace

using OVPropertiesTestsMismatchesNPU = OVPropertiesTestsNPU;
using OVSetRunTimePropertiesTestsNPU = OVPropertiesTestsNPU;
using OVSetCompileTimePropertiesTestsNPU = OVPropertiesTestsNPU;
using OVSetBothPropertiesTestsNPU = OVPropertiesTestsNPU;
using OVSetCartesianProductPropertiesTestsNPU = OVPropertiesTestsNPU;
using OVRunTimePropertiesArgumentsTestsNPU = OVPropertiesArgumentsTestsNPU;
using OVCompileTimePropertiesArgumentsTestsNPU = OVPropertiesArgumentsTestsNPU;
using OVBothPropertiesArgumentsTestsNPU = OVPropertiesArgumentsTestsNPU;

namespace ov::test::behavior {

TEST_P(OVPropertiesTestsMismatchesNPU, DetectPotentialPropertyMismatches) {
    // Update properties' list with compiler properties
    core->get_property(target_device, ov::supported_properties);
    auto registeredProperties = core->get_property(target_device, ov::intel_npu::registered_properties);
    auto propertiesVector = properties.begin()->second.as<std::vector<ov::AnyMap>>();
    std::vector<ov::PropertyName> propertyNames, mismatchingProperties;
    for (const auto& prop : propertiesVector) {
        propertyNames.push_back(prop.begin()->first);
    }

    for (const auto& registeredProperty : registeredProperties) {
        if (registeredProperty.find("NPUW") != ov::PropertyName::npos) {
            continue;
        }
        if (std::find(propertyNames.begin(), propertyNames.end(), registeredProperty) == propertyNames.end()) {
            mismatchingProperties.push_back(registeredProperty);
        }
    }

    ASSERT_TRUE(mismatchingProperties.empty())
        << "Mismatching properties: " << ov::Any(mismatchingProperties).as<std::string>();
}

TEST_P(OVRunTimePropertiesArgumentsTestsNPU, GetRunTimePropertiesNoCompilerLoad) {
    invalidate_umd_caching();
    // Need to exclude properties that are not registered in plugin properties, e.g. LOADED_FROM_CACHE (compiled_model
    // prop)
    auto registered_properties = core->get_property(target_device, ov::intel_npu::registered_properties);
    if (std::find(registered_properties.begin(), registered_properties.end(), propertyName) ==
        registered_properties.end()) {
        GTEST_SKIP() << propertyName << " was not found within plugin registered properties!";
    }
    OV_ASSERT_NO_THROW(core->get_property(target_device, propertyName, properties));
    if (propertyName == ov::supported_properties.name()) {
        ASSERT_TRUE(is_cid_loaded() && !is_cip_loaded());
    } else if (propertyName == ov::intel_npu::compiler_version.name()) {
        ASSERT_TRUE(is_cid_loaded() || is_cip_loaded());
    } else {
        ASSERT_TRUE(!is_cid_loaded() && !is_cip_loaded());
    }
}

TEST_P(OVSetRunTimePropertiesTestsNPU, SetRunTimePropertiesNoCompilerLoad) {
    invalidate_umd_caching();
    // Need to exclude properties that are not registered in plugin properties, e.g. LOADED_FROM_CACHE (compiled_model
    // prop)
    auto registered_properties = core->get_property(target_device, ov::intel_npu::registered_properties);
    if (std::find(registered_properties.begin(), registered_properties.end(), properties.begin()->first) ==
        registered_properties.end()) {
        GTEST_SKIP() << properties.begin()->first << " was not found within plugin registered properties!";
    }
    core->get_property(target_device, ov::available_devices);
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));
    ASSERT_TRUE(!is_cid_loaded() && !is_cip_loaded());
}

TEST_P(OVCompileTimePropertiesArgumentsTestsNPU, GetCompileTimePropertiesCompilerLoad) {
    invalidate_umd_caching();
    OV_ASSERT_NO_THROW(core->get_property(target_device, propertyName, properties));
    ASSERT_TRUE(is_cid_loaded() || is_cip_loaded());
}

TEST_P(OVSetCompileTimePropertiesTestsNPU, SetCompileTimePropertiesCompilerLoad) {
    invalidate_umd_caching();
    core->get_property(target_device, ov::available_devices);
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));
    ASSERT_TRUE(is_cid_loaded() || is_cip_loaded());
}

TEST_P(OVBothPropertiesArgumentsTestsNPU, GetBothPropertiesCompilerLoad) {
    invalidate_umd_caching();
    core->get_property(target_device, ov::available_devices);
    OV_ASSERT_NO_THROW(core->get_property(target_device, propertyName, properties));
    ASSERT_TRUE(is_cid_loaded() || is_cip_loaded());
}

TEST_P(OVSetBothPropertiesTestsNPU, SetBothPropertiesCompilerLoad) {
    invalidate_umd_caching();
    core->get_property(target_device, ov::available_devices);
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));
    ASSERT_TRUE(is_cid_loaded() || is_cip_loaded());
}

TEST_P(OVSetCartesianProductPropertiesTestsNPU, SetCartesianProducthPropertiesCompilerLoad) {
    invalidate_umd_caching();
    core->get_property(target_device, ov::available_devices);
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));
    ASSERT_TRUE(is_cid_loaded() || is_cip_loaded());
}

}  // namespace ov::test::behavior

namespace {
template <typename T>
constexpr std::vector<T> operator+(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    std::vector<T> result;
    result.insert(result.end(), vec1.begin(), vec1.end());
    result.insert(result.end(), vec2.begin(), vec2.end());
    return result;
}

const std::vector<ov::AnyMap> runTimeProperties = {
    {{ov::cache_dir.name(), ""}},
    {{ov::hint::compiled_blob.name(), ov::Tensor()}},
    {{ov::loaded_from_cache.name(), true}},
    {{ov::internal::exclusive_async_requests.name(), true}},
    {{ov::intel_npu::profiling_type.name(), ov::intel_npu::ProfilingType::INFER}},
    {{ov::hint::model_priority.name(), ov::hint::Priority::HIGH}},
    {{ov::intel_npu::create_executor.name(), 3}},
    {{ov::intel_npu::defer_weights_load.name(), true}},
    {{ov::weights_path.name(), ""}},
    {{ov::hint::model.name(), std::shared_ptr<ov::Model>(nullptr)}},
    {{ov::hint::model.name(), std::shared_ptr<const ov::Model>(nullptr)}},
    {{ov::num_streams.name(), ov::streams::Num(2)}},
    {{ov::hint::enable_cpu_pinning.name(), true}},
    {{ov::workload_type.name(), ov::WorkloadType::EFFICIENT}},
    {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::MLIR}},
    {{ov::intel_npu::bypass_umd_caching.name(), true}},
    {{ov::intel_npu::run_inferences_sequentially.name(), true}},
    {{ov::intel_npu::disable_version_check.name(), true}},
};

const std::vector<ov::AnyMap> compileTimeProperties = {
    {{ov::cache_mode.name(), ov::CacheMode::OPTIMIZE_SIZE}},
    {{ov::intel_npu::batch_mode.name(), ov::intel_npu::BatchMode::PLUGIN}},
    {{ov::intel_npu::dma_engines.name(), 1}},
    {{ov::intel_npu::compilation_mode.name(), ""}},
    {{ov::hint::execution_mode.name(), ov::hint::ExecutionMode::ACCURACY}},
    {{ov::intel_npu::dynamic_shape_to_static.name(), true}},
    {{ov::intel_npu::compilation_mode_params.name(), ""}},
    {{ov::intel_npu::tiles.name(), 1}},
    {{ov::intel_npu::stepping.name(), 1}},
    {{ov::intel_npu::max_tiles.name(), 2}},
    {{ov::intel_npu::dma_engines.name(), 2}},
    {{ov::intel_npu::backend_compilation_params.name(), ""}},
    {{ov::compilation_num_threads.name(), 3}},
    {{ov::intel_npu::compiler_dynamic_quantization.name(), true}},
    {{ov::intel_npu::qdq_optimization.name(), true}},
    {{ov::intel_npu::qdq_optimization_aggressive.name(), true}},
    {{ov::intel_npu::batch_compiler_mode_settings.name(), true}},
    {{ov::intel_npu::weightless_blob.name(), true}},
    {{ov::intel_npu::separate_weights_version.name(), ov::intel_npu::WSVersion::ONE_SHOT}},
    {{ov::intel_npu::ws_compile_call_number.name(), 2}},
};

const std::vector<ov::AnyMap> bothProperties = {
    {{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}},
    {{ov::hint::num_requests.name(), 3}},
    {{ov::hint::inference_precision.name(), ov::element::i8}},
    {{ov::enable_profiling.name(), true}},
    {{ov::log::level.name(), ov::log::Level::INFO}},
    {{ov::intel_npu::platform.name(), ov::intel_npu::Platform::AUTO_DETECT}},
    {{ov::device::id.name(), "NPU.0"}},
    {{ov::intel_npu::turbo.name(), true}},
};

const std::vector<ov::AnyMap> metrics = {
    {{ov::available_devices.name(), "NPU"}},
    {{ov::device::architecture.name(), "3720"}},
    {{ov::device::gops.name(),
      std::map<ov::element::Type, float>{{ov::element::bf16, .0f},
                                         {ov::element::f16, 5734.39990234375f},
                                         {ov::element::i8, 11468.7998046875f},
                                         {ov::element::u8, 11468.7998046875f}}}},
    {{ov::device::luid.name(),
      ov::device::LUID{/* .MAX_LUID_SIZE = 8, */ /* .luid  = */ std::
                           array<uint8_t, /* MAX_LUID_SIZE = */ 8>{0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89}}}},
    {{ov::device::pci_info.name(),
      ov::device::PCIInfo{/* .domain = */ 0xFFEEDDCC,
                          /* .bus = */ 0xBBAA9988,
                          /* .device = */ 0x77665544,
                          /* .function = */ 0x33221100}}},
    {{ov::device::type.name(), ov::device::Type::DISCRETE}},
    {{ov::internal::supported_properties.name(), ""}},
    {{ov::device::uuid.name(),
      ov::device::UUID{/* .MAX_UUID_SIZE = 16, */ /* .uuid  = */ std::array<uint8_t, /* MAX_UUID_SIZE = */ 16>{0xAB,
                                                                                                               0xCD,
                                                                                                               0xEF,
                                                                                                               0x01,
                                                                                                               0x23,
                                                                                                               0x45,
                                                                                                               0x67,
                                                                                                               0x89,
                                                                                                               0x98,
                                                                                                               0x76,
                                                                                                               0x54,
                                                                                                               0x32,
                                                                                                               0x10,
                                                                                                               0xFE,
                                                                                                               0xDC,
                                                                                                               0xBA}}}},
    {{ov::execution_devices.name(), std::vector<std::string>{"NPU"}}},
    {{ov::device::full_name.name(), "Intel(R) AI Boost"}},
    {{ov::intel_npu::backend_name.name(), "LEVEL0"}},
    {{ov::intel_npu::compiler_version.name(), 123456}},
    {{ov::internal::caching_properties.name(), ""}},
    {{ov::intel_npu::device_alloc_mem_size.name(), 0}},
    {{ov::intel_npu::device_total_mem_size.name(), 16 * 1024 * 1024}},
    {{ov::intel_npu::driver_version.name(), 1688}},
    {{ov::intel_npu::registered_properties.name(), ""}},
    {{ov::optimal_number_of_infer_requests.name(), 1}},
    {{ov::device::capabilities.name(), std::vector<std::string>{"FP16", "INT8", "EXPORT_IMPORT"}}},
    {{ov::range_for_async_infer_requests.name(), std::tuple<unsigned int, unsigned int, unsigned int>{1, 10, 1}}},
    {{ov::range_for_streams.name(), std::tuple<unsigned int, unsigned int>{1, 4}}},
    {{ov::supported_properties.name(), ""}},
};

const std::vector<ov::AnyMap> allProperties = {
    {{"",
      runTimeProperties + compileTimeProperties + bothProperties + metrics}},  // std::vector<std::vector<ov::AnyMap>>
};

std::vector<std::string> convertToStringVec(const std::vector<ov::AnyMap>& anyMapVec) {
    std::vector<std::string> retVec;
    for (const auto& anyMap : anyMapVec) {
        retVec.push_back(anyMap.begin()->first);
    }
    return retVec;
}

std::vector<ov::AnyMap> CartesianProductAnyMapVec(const std::vector<ov::AnyMap>& anyMapVec1,
                                                  const std::vector<ov::AnyMap>& anyMapVec2,
                                                  const std::vector<ov::AnyMap>& anyMapVec3) {
    std::vector<ov::AnyMap> retAnyMapVec = {};
    for (const auto& anyMap1 : anyMapVec1) {
        for (const auto& anyMap2 : anyMapVec2) {
            for (const auto& anyMap3 : anyMapVec3) {
                retAnyMapVec.push_back({*anyMap1.begin(), *anyMap2.begin(), *anyMap3.begin()});
            }
        }
    }
    return retAnyMapVec;
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesTestsMismatchesNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(allProperties)),
                         ov::test::utils::appendPlatformTypeTestName<OVPropertiesTestsMismatchesNPU>);

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    OVRunTimePropertiesArgumentsTestsNPU,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(convertToStringVec(runTimeProperties) + convertToStringVec(metrics)),
                       ::testing::ValuesIn(runTimeProperties + compileTimeProperties + bothProperties)),
    (ov::test::utils::appendPlatformTypeTestName<OVRunTimePropertiesArgumentsTestsNPU, true>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVSetRunTimePropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(runTimeProperties)),
                         (ov::test::utils::appendPlatformTypeTestName<OVSetRunTimePropertiesTestsNPU, true>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompileTimePropertiesArgumentsTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(convertToStringVec(compileTimeProperties)),
                                            ::testing::ValuesIn(runTimeProperties + compileTimeProperties +
                                                                bothProperties)),
                         (ov::test::utils::appendPlatformTypeTestName<OVCompileTimePropertiesArgumentsTestsNPU, true>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVSetCompileTimePropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compileTimeProperties)),
                         ov::test::utils::appendPlatformTypeTestName<OVSetCompileTimePropertiesTestsNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVBothPropertiesArgumentsTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(convertToStringVec(bothProperties)),
                                            ::testing::ValuesIn(runTimeProperties + compileTimeProperties +
                                                                bothProperties)),
                         (ov::test::utils::appendPlatformTypeTestName<OVBothPropertiesArgumentsTestsNPU, true>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVSetBothPropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(bothProperties)),
                         ov::test::utils::appendPlatformTypeTestName<OVSetBothPropertiesTestsNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVSetCartesianProductPropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(CartesianProductAnyMapVec(runTimeProperties,
                                                                                          compileTimeProperties,
                                                                                          bothProperties))),
                         (ov::test::utils::appendPlatformTypeTestName<OVSetCartesianProductPropertiesTestsNPU, true>));

}  // namespace
