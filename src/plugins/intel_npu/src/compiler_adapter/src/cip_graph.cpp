// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cip_graph.hpp"

#include "intel_npu/config/runtime.hpp"

namespace intel_npu {

CipGraph::CipGraph(const std::shared_ptr<IZeroAdapter>& zeroAdapter,
                   const ov::SoPtr<ICompiler>& compiler,
                   ze_graph_handle_t graphHandle,
                   NetworkMetadata metadata,
                   const std::vector<uint8_t> compiledNetwork,
                   const Config& config)
    : IGraph(graphHandle, std::move(metadata)),
      _zeroAdapter(zeroAdapter),
      _compiler(compiler),
      _compiledNetwork(std::move(compiledNetwork)),
      _config(config),
      _logger("CipGraph", _config.get<LOG_LEVEL>()) {
    if (_config.get<CREATE_EXECUTOR>()) {
        initialize();
    } else {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
    }
}

CompiledNetwork CipGraph::export_blob() const {
    return CompiledNetwork(_compiledNetwork.data(), _compiledNetwork.size(), _compiledNetwork);
}

std::vector<ov::ProfilingInfo> CipGraph::process_profiling_output(const std::vector<uint8_t>& profData) const {
    return _compiler->process_profiling_output(profData, _compiledNetwork, _config);
}

void CipGraph::set_argument_value(uint32_t argi, const void* argv) const {
    if (_zeroAdapter == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _zeroAdapter->setArgumentValue(_handle, argi, argv);
}

void CipGraph::initialize() {
    if (_zeroAdapter) {
        _logger.debug("Graph initialize start");

        std::tie(_input_descriptors, _output_descriptors) = _zeroAdapter->getIODesc(_handle);
        _command_queue = _zeroAdapter->crateCommandQueue(_config);

        if (_config.has<WORKLOAD_TYPE>()) {
            setWorkloadType(_config.get<WORKLOAD_TYPE>());
        }

        _zeroAdapter->graphInitialie(_handle, _config);

        _logger.debug("Graph initialize finish");
    }
}

CipGraph::~CipGraph() {
    if (_handle != nullptr) {
        auto result = _zeroAdapter->release(_handle);

        if (ZE_RESULT_SUCCESS == result) {
            _handle = nullptr;
        }
    }
}

}  // namespace intel_npu
