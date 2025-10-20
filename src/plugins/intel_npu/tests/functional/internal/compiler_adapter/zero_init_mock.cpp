// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_init_mock.hpp"

#include <ze_command_queue_npu_ext.h>
#include <ze_mem_import_system_memory_ext.h>

#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {

const ze_driver_uuid_t ZeroInitStructsMock::uuid = ze_intel_npu_driver_uuid;

static std::tuple<uint32_t, std::string> queryDriverExtensionVersion(
    const char* extName,
    uint32_t extCurrentVersion,
    std::vector<ze_driver_extension_properties_t>& extProps,
    uint32_t count) {
    const char* functionExtName = nullptr;
    uint32_t targetVersion = 0;

    for (uint32_t i = 0; i < count; ++i) {
        auto& property = extProps[i];

        if (strncmp(property.name, extName, strlen(extName)) != 0) {
            continue;
        }

        if (property.version >= extCurrentVersion) {
            functionExtName = property.name;
            targetVersion = extCurrentVersion;
            break;
        }

        // Use the latest version supported by the driver - We need to go through all the properties for older drivers
        // that use specific names for different graph ext versions, e.g.: ZE_extension_graph_1_1,
        // ZE_extension_graph_1_2
        if (property.version > targetVersion) {
            functionExtName = property.name;
            targetVersion = property.version;
        }
    }

    return std::make_tuple(targetVersion, functionExtName ? functionExtName : "");
}

void ZeroInitStructsMock::initNpuDriver() {
    auto setNpuDriver = [&](uint32_t drivers_count, std::vector<ze_driver_handle_t> all_drivers) {
        driver_properties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
        log.debug("ZeroInitStructsMock::initNpuDriver - setting driver properties to "
                  "ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES");
        for (uint32_t i = 0; i < drivers_count; ++i) {
            zeDriverGetProperties(all_drivers[i], &driver_properties);

            if (memcmp(&driver_properties.uuid, &uuid, sizeof(uuid)) == 0) {
                driver_handle = all_drivers[i];
                break;
            }
        }
        if (driver_handle == nullptr) {
            OPENVINO_THROW("NPU driver wasn't found!");
        }
    };

    auto fallbackToZeDriverGet = [&]() {
        log.debug("ZeroInitStructsMock - zeInitDrivers not supported, fallback to zeDriverGet");

        uint32_t drivers_count = 0;
        THROW_ON_FAIL_FOR_LEVELZERO("zeDriverGet", zeDriverGet(&drivers_count, nullptr));

        std::vector<ze_driver_handle_t> all_drivers(drivers_count);
        THROW_ON_FAIL_FOR_LEVELZERO("zeDriverGet", zeDriverGet(&drivers_count, all_drivers.data()));

        // Get our target driver
        setNpuDriver(drivers_count, std::move(all_drivers));
    };

    zel_version_t loader_version = {};
    size_t num_components;
    auto result = zelLoaderGetVersions(&num_components, nullptr);
    if (result == ZE_RESULT_SUCCESS) {
        zel_component_version_t* versions = new zel_component_version_t[num_components];
        result = zelLoaderGetVersions(&num_components, versions);

        if (result == ZE_RESULT_SUCCESS) {
            for (size_t i = 0; i < num_components; ++i) {
                if (strncmp(versions[i].component_name, "loader", strlen("loader")) == 0) {
                    loader_version = versions[i].component_lib_version;

                    log.debug("ZeroInitStructsMock - ze_loader.dll version: %d.%d.%d",
                              loader_version.major,
                              loader_version.minor,
                              loader_version.patch);
                }
            }
        }

        delete[] versions;
    }

    if (loader_version.major > 1 || (loader_version.major == 1 && loader_version.minor > 18) ||
        (loader_version.major == 1 && loader_version.minor == 18 && loader_version.patch >= 5)) {
        uint32_t drivers_count = 0;
        ze_init_driver_type_desc_t desc = {};
        desc.flags = ZE_INIT_DRIVER_TYPE_FLAG_NPU;
        auto result = zeInitDrivers(&drivers_count, nullptr, &desc);
        if (result != ZE_RESULT_SUCCESS) {
            fallbackToZeDriverGet();
            return;
        }

        std::vector<ze_driver_handle_t> all_drivers(drivers_count);
        result = zeInitDrivers(&drivers_count, all_drivers.data(), &desc);
        if (result != ZE_RESULT_SUCCESS) {
            fallbackToZeDriverGet();
            return;
        }

        // Get our target driver
        setNpuDriver(drivers_count, std::move(all_drivers));

        return;
    }

    fallbackToZeDriverGet();
}

ZeroInitStructsMock::ZeroInitStructsMock(int extVersion)
    : zero_api(ZeroApi::getInstance()),
      log("NPUZeroInitStructsMock", Logger::global().level()) {
    log.debug("ZeroInitStructsMock - performing zeInit on NPU only");
    THROW_ON_FAIL_FOR_LEVELZERO("zeInit", zeInit(ZE_INIT_FLAG_VPU_ONLY));

    log.debug("ZeroInitStructsMock - initialize NPU Driver");
    initNpuDriver();

    // Check L0 API version
    THROW_ON_FAIL_FOR_LEVELZERO("zeDriverGetApiVersion", zeDriverGetApiVersion(driver_handle, &ze_drv_api_version));

    if (ZE_MAJOR_VERSION(ZE_API_VERSION_CURRENT) != ZE_MAJOR_VERSION(ze_drv_api_version)) {
        OPENVINO_THROW("Incompatibility between NPU plugin and driver! ",
                       "Plugin L0 API major version = ",
                       ZE_MAJOR_VERSION(ZE_API_VERSION_CURRENT),
                       ", ",
                       "Driver L0 API major version = ",
                       ZE_MAJOR_VERSION(ze_drv_api_version));
    }
    if (ZE_MINOR_VERSION(ZE_API_VERSION_CURRENT) != ZE_MINOR_VERSION(ze_drv_api_version)) {
        log.warning("Some features might not be available! "
                    "Plugin L0 API minor version = %d, Driver L0 API minor version = %d",
                    ZE_MINOR_VERSION(ZE_API_VERSION_CURRENT),
                    ZE_MINOR_VERSION(ze_drv_api_version));
    }

    uint32_t count = 0;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDriverGetExtensionProperties",
                                zeDriverGetExtensionProperties(driver_handle, &count, nullptr));

    std::vector<ze_driver_extension_properties_t> extProps;
    extProps.resize(count);
    THROW_ON_FAIL_FOR_LEVELZERO("zeDriverGetExtensionProperties",
                                zeDriverGetExtensionProperties(driver_handle, &count, extProps.data()));

    // save the list of extension properties for later searches
    for (auto it = extProps.begin(); it != extProps.end(); ++it) {
        ze_driver_extension_properties_t p = *it;
        driver_extension_properties.emplace(std::string(p.name), p.version);
    }

    // Query our graph extension version
    std::string graph_ext_name;
    uint32_t graph_ext_version = 0;
    uint32_t target_graph_ext_version = extVersion;

    log.debug("Try to find graph ext version: %d.%d",
              ZE_MAJOR_VERSION(target_graph_ext_version),
              ZE_MINOR_VERSION(target_graph_ext_version));
    std::tie(graph_ext_version, graph_ext_name) =
        queryDriverExtensionVersion(ZE_GRAPH_EXT_NAME, target_graph_ext_version, extProps, count);

    if (graph_ext_name.empty()) {
        OPENVINO_THROW("queryGraphExtensionVersion: Failed to find Graph extension in NPU Driver");
    }

    const uint16_t supported_driver_ext_major_version = ZE_MAJOR_VERSION(target_graph_ext_version);
    const uint16_t driver_ext_major_version = ZE_MAJOR_VERSION(graph_ext_version);
    if (supported_driver_ext_major_version != driver_ext_major_version) {
        OPENVINO_THROW("Plugin supports only driver with graph extension major version ",
                       supported_driver_ext_major_version,
                       "; discovered driver graph extension has major version ",
                       driver_ext_major_version);
    }

    log.info("Found Driver Version %d.%d, Graph Extension Version %d.%d (%s)",
             ZE_MAJOR_VERSION(ze_drv_api_version),
             ZE_MINOR_VERSION(ze_drv_api_version),
             ZE_MAJOR_VERSION(graph_ext_version),
             ZE_MINOR_VERSION(graph_ext_version),
             graph_ext_name.c_str());

    // Query our command queue extension version
    std::string command_queue_ext_name;
    uint32_t command_queue_ext_version = 0;
    std::tie(command_queue_ext_version, command_queue_ext_name) =
        queryDriverExtensionVersion(ZE_COMMAND_QUEUE_NPU_EXT_NAME,
                                    ZE_COMMAND_QUEUE_NPU_EXT_VERSION_CURRENT,
                                    extProps,
                                    count);

    log.debug("NPU command queue version %d.%d",
              ZE_MAJOR_VERSION(command_queue_ext_version),
              ZE_MINOR_VERSION(command_queue_ext_version));

    // Load our command queue extension
    ze_command_queue_npu_dditable_ext_last_t* _command_queue_npu_dditable_ext = nullptr;
    if (command_queue_ext_version) {
        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeDriverGetExtensionFunctionAddress " + command_queue_ext_name,
            zeDriverGetExtensionFunctionAddress(driver_handle,
                                                command_queue_ext_name.c_str(),
                                                reinterpret_cast<void**>(&_command_queue_npu_dditable_ext)));
    }

    command_queue_npu_dditable_ext_decorator =
        std::make_unique<ze_command_queue_npu_dditable_ext_decorator>(_command_queue_npu_dditable_ext,
                                                                      command_queue_ext_version);

    // Load our graph extension
    ze_graph_dditable_ext_t* graph_ddi_table_ext = nullptr;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDriverGetExtensionFunctionAddress",
                                zeDriverGetExtensionFunctionAddress(driver_handle,
                                                                    graph_ext_name.c_str(),
                                                                    reinterpret_cast<void**>(&graph_ddi_table_ext)));
    graph_dditable_ext_decorator =
        std::make_unique<ze_graph_dditable_ext_decorator>(graph_ddi_table_ext, graph_ext_version);

    // Query the mutable command list version
    [[maybe_unused]] std::string mutuable_command_list_ext_name;
    std::tie(mutable_command_list_ext_version, mutuable_command_list_ext_name) =
        queryDriverExtensionVersion(ZE_MUTABLE_COMMAND_LIST_EXP_NAME,
                                    ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_CURRENT,
                                    extProps,
                                    count);

    log.debug("Mutable command list version %d.%d",
              ZE_MAJOR_VERSION(mutable_command_list_ext_version),
              ZE_MINOR_VERSION(mutable_command_list_ext_version));

    // Load our profiling extension
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;
    THROW_ON_FAIL_FOR_LEVELZERO(
        "zeDriverGetExtensionFunctionAddress",
        zeDriverGetExtensionFunctionAddress(driver_handle,
                                            "ZE_extension_profiling_data",
                                            reinterpret_cast<void**>(&_graph_profiling_ddi_table_ext)));

    graph_profiling_npu_dditable_ext_decorator =
        std::make_unique<ze_graph_profiling_ddi_table_ext_decorator>(_graph_profiling_ddi_table_ext);

    uint32_t device_count = 1;
    // Get our target device
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGet", zeDeviceGet(driver_handle, &device_count, &device_handle));

    // Create context - share between the compiler and the backend
    ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0, 0};
    THROW_ON_FAIL_FOR_LEVELZERO("zeContextCreate", zeContextCreate(driver_handle, &context_desc, &context));
    log.debug("ZeroInitStructsMock initialize complete");

    // Obtain compiler-in-driver properties
    compiler_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES;
    auto result = graph_dditable_ext_decorator->pfnDeviceGetGraphProperties(device_handle, &compiler_properties);
    THROW_ON_FAIL_FOR_LEVELZERO("pfnDeviceGetGraphProperties", result);

    // Discover if standard allocation is supported
    ze_device_external_memory_properties_t external_memory_properties_desc = {};
    external_memory_properties_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES;
    auto res = zeDeviceGetExternalMemoryProperties(device_handle, &external_memory_properties_desc);
    if (res == ZE_RESULT_SUCCESS) {
#ifdef _WIN32
        if (external_memory_properties_desc.memoryAllocationImportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32) {
            _external_memory_fd_win32_supported = true;
        }
#else
        if (external_memory_properties_desc.memoryAllocationImportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF) {
            _external_memory_fd_win32_supported = true;
        }
#endif

        if (external_memory_properties_desc.memoryAllocationImportTypes &
            ZE_EXTERNAL_MEMORY_TYPE_FLAG_STANDARD_ALLOCATION) {
            _external_memory_standard_allocation_supported = true;
        }
    }
}

ZeroInitStructsMock::~ZeroInitStructsMock() {
    if (context) {
        log.debug("ZeroInitStructsMock - performing zeContextDestroy");
        auto result = zeContextDestroy(context);
        if (result != ZE_RESULT_SUCCESS) {
            if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
                log.warning("zeContextDestroy failed to destroy the context; Level zero context was already destroyed");
            } else {
                log.error("zeContextDestroy failed %#X", uint64_t(result));
            }
        }
    }
}

}  // namespace intel_npu
