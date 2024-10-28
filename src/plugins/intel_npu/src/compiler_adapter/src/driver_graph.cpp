// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_graph.hpp"

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/runtime.hpp"

namespace intel_npu {

DriverGraph::DriverGraph(const std::shared_ptr<IZeroAdapter>& adapter,
                         ze_graph_handle_t graphHandle,
                         NetworkMetadata metadata,
                         const Config& config,
                         std::optional<std::vector<uint8_t>> network)
    : IGraph(graphHandle, std::move(metadata)),
      _adapter(adapter),
      _config(config),
      _logger("DriverGraph", _config.get<LOG_LEVEL>()) {
    if (network.has_value()) {
        _networkStorage = std::move(*network);
    }

    if (_config.get<CREATE_EXECUTOR>()) {
        initialize();
    } else {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
    }
}

CompiledNetwork DriverGraph::export_blob() const {
    return _adapter->getCompiledNetwork(_handle);
}

std::vector<ov::ProfilingInfo> DriverGraph::process_profiling_output(const std::vector<uint8_t>& profData) const {
    OPENVINO_THROW("Profiling post-processing is not supported.");
}

void DriverGraph::set_argument_value(uint32_t argi, const void* argv) const {
    if (_adapter == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _adapter->setArgumentValue(_handle, argi, argv);
}

void DriverGraph::initialize() {
    if (_adapter) {
        _logger.debug("Graph initialize start");

        std::tie(_input_descriptors, _output_descriptors) = _adapter->getIODesc(_handle);
        _command_queue = _adapter->crateCommandQueue(_config);

        if (_config.has<WORKLOAD_TYPE>()) {
            set_workload_type(_config.get<WORKLOAD_TYPE>());
        }

        _adapter->graphInitialize(_handle, _config);

        _logger.debug("Graph initialize finish");
    }
}

DriverGraph::~DriverGraph() {
    if (_handle != nullptr) {
        auto result = _adapter->release(_handle);

        if (ZE_RESULT_SUCCESS == result) {
            _handle = nullptr;
        }
    }
}

}  // namespace intel_npu
