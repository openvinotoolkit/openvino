// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_init.hpp"

#include <ze_command_queue_npu_ext.h>

#include <regex>

#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

#ifdef _WIN32
namespace {
constexpr uint32_t WIN_DRIVER_NO_MCL_SUPPORT = 2688;
}  // namespace

#endif

namespace intel_npu {

const ze_driver_uuid_t ZeroInitStructsHolder::uuid = ze_intel_npu_driver_uuid;

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

        if (property.version == extCurrentVersion) {
            functionExtName = property.name;
            targetVersion = property.version;
            break;
        }

        // Use the latest version supported by the driver.
        if (property.version > targetVersion) {
            functionExtName = property.name;
            targetVersion = property.version;
        }
    }

    return std::make_tuple(targetVersion, functionExtName ? functionExtName : "");
}

ZeroInitStructsHolder::ZeroInitStructsHolder() : log("NPUZeroInitStructsHolder", Logger::global().level()) {
    log.debug("ZeroInitStructsHolder - performing zeInit on VPU only");
    THROW_ON_FAIL_FOR_LEVELZERO("zeInit", zeInit(ZE_INIT_FLAG_VPU_ONLY));

    uint32_t drivers = 0;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDriverGet", zeDriverGet(&drivers, nullptr));

    std::vector<ze_driver_handle_t> all_drivers(drivers);
    THROW_ON_FAIL_FOR_LEVELZERO("zeDriverGet", zeDriverGet(&drivers, all_drivers.data()));

    // Get our target driver
    driver_properties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
    log.debug("ZeroInitStructsHolder - setting driver properties to ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES");
    for (uint32_t i = 0; i < drivers; ++i) {
        zeDriverGetProperties(all_drivers[i], &driver_properties);

        if (memcmp(&driver_properties.uuid, &uuid, sizeof(uuid)) == 0) {
            driver_handle = all_drivers[i];
            break;
        }
    }
    if (driver_handle == nullptr) {
        OPENVINO_THROW("zeDriverGet failed to return NPU driver");
    }

    // Check L0 API version
    ze_api_version_t ze_drv_api_version = {};
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
    uint32_t target_graph_ext_version = ZE_GRAPH_EXT_VERSION_CURRENT;

#if defined(NPU_PLUGIN_DEVELOPER_BUILD)
    const char* extVersion = std::getenv("NPU_ZE_GRAPH_EXT_VERSION");
    if (extVersion) {
        std::string extVersionString(extVersion);
        std::regex extVersionRegex(R"(^(\d+)\.(\d+)$)");
        std::smatch match;

        if (std::regex_match(extVersionString, match, extVersionRegex)) {
            int major = std::stoi(match[1].str());
            int minor = std::stoi(match[2].str());
            log.debug("Try to find graph ext version: %d.%d instead of %d.%d.",
                      major,
                      minor,
                      ZE_MAJOR_VERSION(target_graph_ext_version),
                      ZE_MINOR_VERSION(target_graph_ext_version));
            target_graph_ext_version = ZE_MAKE_VERSION(major, minor);
        }
    }
#endif

    log.debug("Try to find graph ext version: %d.%d",
              ZE_MAJOR_VERSION(target_graph_ext_version),
              ZE_MINOR_VERSION(target_graph_ext_version));
    std::tie(graph_ext_version, graph_ext_name) =
        queryDriverExtensionVersion(ZE_GRAPH_EXT_NAME, target_graph_ext_version, extProps, count);

    if (graph_ext_name.empty()) {
        OPENVINO_THROW("queryGraphExtensionVersion: Failed to find Graph extension in NPU Driver");
    }

    // Use version that plugin can support as identifier to control workflow
    if (graph_ext_version > target_graph_ext_version) {
        log.warning("Graph extension version from driver is %d.%d. "
                    "Larger than plugin max graph ext version %d.%d. "
                    "Force to use plugin ext version with the new table to control flow!",
                    ZE_MAJOR_VERSION(graph_ext_version),
                    ZE_MINOR_VERSION(graph_ext_version),
                    ZE_MAJOR_VERSION(target_graph_ext_version),
                    ZE_MINOR_VERSION(target_graph_ext_version));
        graph_ext_version = target_graph_ext_version;
    }

    const uint16_t supported_driver_ext_major_version = 1;
    const uint16_t driver_ext_major_version = ZE_MAJOR_VERSION(graph_ext_version);
    if (supported_driver_ext_major_version != driver_ext_major_version) {
        OPENVINO_THROW("Plugin supports only driver with extension major version ",
                       supported_driver_ext_major_version,
                       "; discovered driver extension has major version ",
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
#ifdef _WIN32
    // The 2688 Windows driver version doesn't support as expected the MutableCommandList feature
    if (driver_properties.driverVersion != WIN_DRIVER_NO_MCL_SUPPORT) {
#endif
        [[maybe_unused]] std::string mutuable_command_list_ext_name;
        std::tie(mutable_command_list_version, mutuable_command_list_ext_name) =
            queryDriverExtensionVersion(ZE_MUTABLE_COMMAND_LIST_EXP_NAME,
                                        ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_CURRENT,
                                        extProps,
                                        count);
#ifdef _WIN32
    }
#endif

    log.debug("Mutable command list version %d.%d",
              ZE_MAJOR_VERSION(mutable_command_list_version),
              ZE_MINOR_VERSION(mutable_command_list_version));

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
    log.debug("ZeroInitStructsHolder initialize complete");
}

ZeroInitStructsHolder::~ZeroInitStructsHolder() {
    if (context) {
        log.debug("ZeroInitStructsHolder - performing zeContextDestroy");
        auto result = zeContextDestroy(context);
        if (ZE_RESULT_SUCCESS != result) {
            log.error("zeContextDestroy failed %#X", uint64_t(result));
        }
    }
}

}  // namespace intel_npu
