// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compiler_adapter.hpp"

#include "backends.hpp"
#include "graph_transformations.hpp"
#include "intel_npu/al/config/common.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "ze_intel_npu_uuid.h"
#include "zero_backend.hpp"
#include "zero_compiler_in_driver.hpp"
#include "zero_init.hpp"

namespace intel_npu {
namespace driverCompilerAdapter {

LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(std::shared_ptr<NPUBackends> npuBackends)
    : _logger("LevelZeroCompilerAdapter", Logger::global().level()) {
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
    std::shared_ptr<intel_npu::IEngineBackend> iEngineBackend = soPtrBackend._ptr;  // Extract the raw pointer
    std::shared_ptr<ZeroEngineBackend> zeroBackend = nullptr;
    try {
        zeroBackend = std::dynamic_pointer_cast<ZeroEngineBackend>(iEngineBackend);
        if (!zeroBackend) {
            OPENVINO_THROW("LevelZeroCompilerAdapter init failed to cast zeroBackend, zeroBackend is a nullptr");
        }
    } catch (const std::exception& e) {
        OPENVINO_THROW("LevelZeroCompilerAdapter init failed to cast zeroBackend");
    }

    ze_context_handle_t zeContext = (ze_context_handle_t)zeroBackend->getContext();
    ze_driver_handle_t driverHandle = (ze_driver_handle_t)zeroBackend->getDriverHandle();
    ze_device_handle_t deviceHandle = (ze_device_handle_t)zeroBackend->getDeviceHandle();

    uint32_t targetVersion = zeroBackend->getTargetVersion();
    char* graphExtName = zeroBackend->getGraphExtName();

    if (driverHandle == nullptr) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: Failed to get properties about zeDriver");
        return;
    }

    switch (targetVersion) {
    case ZE_GRAPH_EXT_VERSION_1_3:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_3_t>>(graphExtName,
                                                                                              driverHandle,
                                                                                              deviceHandle,
                                                                                              zeContext);
        break;
    case ZE_GRAPH_EXT_VERSION_1_4:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_4_t>>(graphExtName,
                                                                                              driverHandle,
                                                                                              deviceHandle,
                                                                                              zeContext);
        break;
    case ZE_GRAPH_EXT_VERSION_1_5:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_5_t>>(graphExtName,
                                                                                              driverHandle,
                                                                                              deviceHandle,
                                                                                              zeContext);
        break;
    case ZE_GRAPH_EXT_VERSION_1_6:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_6_t>>(graphExtName,
                                                                                              driverHandle,
                                                                                              deviceHandle,
                                                                                              zeContext);
        break;
    default:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_2_t>>(graphExtName,
                                                                                              driverHandle,
                                                                                              deviceHandle,
                                                                                              zeContext);
        break;
    }
    _logger.info("initialize LevelZeroCompilerAdapter complete, using ext_version :  %s", targetVersion);
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
