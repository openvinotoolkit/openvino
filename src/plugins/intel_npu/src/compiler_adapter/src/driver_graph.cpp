// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_graph.hpp"

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"

namespace intel_npu {

DriverGraph::DriverGraph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
                         const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                         ze_graph_handle_t graphHandle,
                         NetworkMetadata metadata,
                         const Config& config,
                         std::optional<std::vector<uint8_t>> blob)
    : IGraph(graphHandle, std::move(metadata), std::move(blob)),
      _zeGraphExt(zeGraphExt),
      _zeroInitStruct(zeroInitStruct),
      _logger("DriverGraph", config.get<LOG_LEVEL>()) {
    if (_zeGraphExt == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }

    if (!config.get<CREATE_EXECUTOR>() || config.get<DEFER_WEIGHTS_LOAD>()) {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
        return;
    }

    initialize(config);
}

void DriverGraph::export_blob(std::ostream& stream) const {
    const uint8_t* blobPtr = nullptr;
    size_t blobSize = -1;
    std::vector<uint8_t> blob;

    if (_blobIsReleased) {
        OPENVINO_THROW("Model was imported (not compiled) by the plugin. Model export is forbidden in this case!");
    }

    _zeGraphExt->getGraphBinary(_handle, blob, blobPtr, blobSize);

    stream.write(reinterpret_cast<const char*>(blobPtr), blobSize);

    if (!stream) {
        _logger.error("Write blob to stream failed. Blob is broken!");
        return;
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        std::uint32_t result = 1171117u;
        for (const uint8_t* it = blobPtr; it != blobPtr + blobSize; ++it) {
            result = ((result << 7) + result) + static_cast<uint32_t>(*it);
        }

        std::stringstream str;
        str << "Blob size: " << blobSize << ", hash: " << std::hex << result;
        _logger.info(str.str().c_str());
    }
    _logger.info("Write blob to stream successfully.");
}

std::vector<ov::ProfilingInfo> DriverGraph::process_profiling_output(const std::vector<uint8_t>& profData,
                                                                     const Config& config) const {
    OPENVINO_THROW("Profiling post-processing is not supported.");
}

void DriverGraph::set_argument_value(uint32_t argi, const void* argv) const {
    _zeGraphExt->setGraphArgumentValue(_handle, argi, argv);
}

void DriverGraph::initialize(const Config& config) {
    _logger.debug("Graph initialize start");

    _logger.debug("performing pfnGetProperties");
    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    auto result = _zeroInitStruct->getGraphDdiTable().pfnGetProperties(_handle, &props);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetProperties", result, _zeroInitStruct->getGraphDdiTable());

    _logger.debug("performing pfnGetArgumentProperties3");
    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        auto result = _zeroInitStruct->getGraphDdiTable().pfnGetArgumentProperties3(_handle, index, &arg3);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _zeroInitStruct->getGraphDdiTable());

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            _input_descriptors.push_back(ArgumentDescriptor{arg3, index});
        } else {
            _output_descriptors.push_back(ArgumentDescriptor{arg3, index});
        }
    }

    _input_descriptors.shrink_to_fit();
    _output_descriptors.shrink_to_fit();

    ze_device_properties_t deviceProperties = {};
    deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                zeDeviceGetProperties(_zeroInitStruct->getDevice(), &deviceProperties));
    auto groupOrdinal = zeroUtils::findGroupOrdinal(_zeroInitStruct->getDevice(), deviceProperties);

    bool turbo = false;
    if (config.has<TURBO>()) {
        turbo = config.get<TURBO>();
    }

    _command_queue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                    zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                    groupOrdinal,
                                                    turbo);

    if (config.has<WORKLOAD_TYPE>()) {
        set_workload_type(config.get<WORKLOAD_TYPE>());
    }

    _zeGraphExt->initializeGraph(_handle, config);

    _logger.debug("Graph initialize finish");

    //  We are allowed to release the original blob because weights were loaded in NPU memory during
    //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
    //  releasing it here to avoid unnecessary memory usage.
    _blobIsReleased = release_blob(config);
}

bool DriverGraph::release_blob(const Config& config) {
    if (_blob.empty() || _zeroInitStruct->getGraphDdiTable().version() < ZE_GRAPH_EXT_VERSION_1_8 ||
        config.get<PERF_COUNT>()) {
        return false;
    }

    ze_graph_properties_2_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    _zeroInitStruct->getGraphDdiTable().pfnGetProperties2(_handle, &properties);

    if (~properties.initStageRequired & ZE_GRAPH_STAGE_INITIALIZE) {
        return false;
    }

    _blob.clear();
    _blob.shrink_to_fit();

    _logger.debug("Blob is released");

    return true;
};

DriverGraph::~DriverGraph() {
    if (_handle != nullptr) {
        auto result = _zeGraphExt->destroyGraph(_handle);

        if (ZE_RESULT_SUCCESS == result) {
            _handle = nullptr;
        }
    }
}

}  // namespace intel_npu
