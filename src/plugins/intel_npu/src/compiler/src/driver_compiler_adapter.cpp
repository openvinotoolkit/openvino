// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compiler_adapter.hpp"

#include "graph_transformations.hpp"
#include "intel_npu/al/config/common.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "ze_intel_npu_uuid.h"
#include "zero_compiler_in_driver.hpp"
#include "zero_init.hpp"
#include "backends.hpp" 
#include "zero_backend.hpp" 

namespace intel_npu {
namespace driverCompilerAdapter {

LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(std::shared_ptr<NPUBackends> npuBackends) : _logger("LevelZeroCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize zeAPI start");
    auto result = zeInit(ZE_INIT_FLAG_VPU_ONLY);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 initialize zeAPI",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    ov::SoPtr<intel_npu::IEngineBackend> soPtrBackend = npuBackends->getIEngineBackend();
    std::shared_ptr<intel_npu::IEngineBackend> iEngineBackend = soPtrBackend._ptr; // Extract the raw pointer
    std::shared_ptr<ZeroEngineBackend> zeroBackend = std::dynamic_pointer_cast<ZeroEngineBackend>(iEngineBackend);

    ze_context_handle_t _context = (ze_context_handle_t)zeroBackend->getContext();
    _driverHandle = (ze_driver_handle_t)zeroBackend->getDriverHandle();
    ze_device_handle_t _deviceHandle = (ze_device_handle_t)zeroBackend->getDeviceHandle();

    if (_driverHandle == nullptr) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: Failed to get properties about zeDriver");
        return;
    }

    // query the extension properties
    uint32_t count = 0;
    result = zeDriverGetExtensionProperties(_driverHandle, &count, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 zeDriverGetExtensionProperties get count",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }
    std::vector<ze_driver_extension_properties_t> extProps;
    extProps.resize(count);
    result = zeDriverGetExtensionProperties(_driverHandle, &count, extProps.data());
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 zeDriverGetExtensionProperties get properties",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }
    const char* graphExtName = nullptr;
    uint32_t targetVersion = 0;
    for (uint32_t i = 0; i < count; ++i) {
        auto& property = extProps[i];

        if (strncmp(property.name, ZE_GRAPH_EXT_NAME, strlen(ZE_GRAPH_EXT_NAME)) != 0) {
            continue;
        }

        // If the driver version is latest, will just use its name.
        if (property.version == ZE_GRAPH_EXT_VERSION_CURRENT) {
            graphExtName = property.name;
            targetVersion = property.version;
            break;
        }

        // Use the latest version supported by the driver.
        if (property.version > targetVersion) {
            graphExtName = property.name;
            targetVersion = property.version;
        }
    }

    if (graphExtName == nullptr) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: Failed to find Graph extension in NPU Driver");
    }

    const uint16_t adapterMajorVersion = 1;
    uint16_t driverMajorVersion = ZE_MAJOR_VERSION(targetVersion);
    if (adapterMajorVersion != driverMajorVersion) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: adapterMajorVersion: ",
                       adapterMajorVersion,
                       " and driverMajorVersion: ",
                       driverMajorVersion,
                       " mismatch!");
    }

#if defined(NPU_PLUGIN_DEVELOPER_BUILD)
    auto adapterManualConfig = std::getenv("ADAPTER_MANUAL_CONFIG");
    if (adapterManualConfig != nullptr) {
        if (strcmp(adapterManualConfig, "ZE_extension_graph_1_6") == 0) {
            _logger.info("With ADAPTER_MANUAL_CONFIG. Using ZE_GRAPH_EXT_VERSION_1_6");
            targetVersion = ZE_GRAPH_EXT_VERSION_1_6;
        } else if (strcmp(adapterManualConfig, "ZE_extension_graph_1_5") == 0) {
            _logger.info("With ADAPTER_MANUAL_CONFIG. Using ZE_GRAPH_EXT_VERSION_1_5");
            targetVersion = ZE_GRAPH_EXT_VERSION_1_5;
        } else if (strcmp(adapterManualConfig, "ZE_extension_graph_1_4") == 0) {
            _logger.info("With ADAPTER_MANUAL_CONFIG. Using ZE_GRAPH_EXT_VERSION_1_4");
            targetVersion = ZE_GRAPH_EXT_VERSION_1_4;
        } else if (strcmp(adapterManualConfig, "ZE_extension_graph_1_3") == 0) {
            _logger.info("With ADAPTER_MANUAL_CONFIG. Using ZE_GRAPH_EXT_VERSION_1_3");
            targetVersion = ZE_GRAPH_EXT_VERSION_1_3;
        } else if (strcmp(adapterManualConfig, "ZE_extension_graph_1_2") == 0) {
            _logger.info("With ADAPTER_MANUAL_CONFIG. Using ZE_GRAPH_EXT_VERSION_1_2");
            targetVersion = ZE_GRAPH_EXT_VERSION_1_2;
        } else {
            OPENVINO_THROW("Using unsupported ADAPTER_MANUAL_CONFIG!");
        }
    }
#endif

    if (ZE_GRAPH_EXT_VERSION_1_3 == targetVersion) {
        _logger.info("Using ZE_GRAPH_EXT_VERSION_1_3");
        apiAdapter =
            std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_3_t>>(graphExtName, _driverHandle, _deviceHandle, _context);
    } else if (ZE_GRAPH_EXT_VERSION_1_4 == targetVersion) {
        _logger.info("Using ZE_GRAPH_EXT_VERSION_1_4");
        apiAdapter =
            std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_4_t>>(graphExtName, _driverHandle, _deviceHandle, _context);
    } else if (ZE_GRAPH_EXT_VERSION_1_5 == targetVersion) {
        _logger.info("Using ZE_GRAPH_EXT_VERSION_1_5");
        apiAdapter =
            std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_5_t>>(graphExtName, _driverHandle, _deviceHandle, _context);
    } else if (ZE_GRAPH_EXT_VERSION_1_6 == targetVersion) {
        _logger.info("Using ZE_GRAPH_EXT_VERSION_1_6");
        apiAdapter =
            std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_6_t>>(graphExtName, _driverHandle, _deviceHandle, _context);
    } else {
        _logger.info("Using ZE_GRAPH_EXT_VERSION_1_2");
        apiAdapter =
            std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_2_t>>(graphExtName, _driverHandle, _deviceHandle, _context);
    }
    _logger.debug("initialize zeAPI end");
}

uint32_t LevelZeroCompilerAdapter::getSupportedOpsetVersion() const {
    return apiAdapter->getSupportedOpsetVersion();
}

NetworkDescription LevelZeroCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                     const Config& config) const {
    _logger.debug("compile start");
    return apiAdapter->compile(model, config);
}

ov::SupportedOpsMap LevelZeroCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                    const Config& config) const {
    _logger.debug("query start");
    return apiAdapter->query(model, config);
}

NetworkMetadata LevelZeroCompilerAdapter::parse(const std::vector<uint8_t>& network, const Config& config) const {
    _logger.debug("parse start");
    return apiAdapter->parse(network, config);
}

std::vector<ov::ProfilingInfo> LevelZeroCompilerAdapter::process_profiling_output(const std::vector<uint8_t>&,
                                                                                  const std::vector<uint8_t>&,
                                                                                  const Config&) const {
    OPENVINO_THROW("Profiling post-processing is not implemented.");
}

}  // namespace driverCompilerAdapter
}  // namespace intel_npu
