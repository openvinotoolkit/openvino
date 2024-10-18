// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cip_graph.hpp"

#include "intel_npu/al/config/runtime.hpp"

namespace intel_npu {

CipGraph::CipGraph(const std::shared_ptr<IZeroLink>& zeroLink,
                   const ov::SoPtr<ICompiler>& compiler,
                   ze_graph_handle_t graphHandle,
                   NetworkMetadata metadata,
                   const std::vector<uint8_t> compiledNetwork,
                   const Config& config)
    : IGraph(graphHandle, std::move(metadata)),
      _zeroLink(zeroLink),
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
    if (_zeroLink == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _zeroLink->setArgumentValue(_handle, argi, argv);
}

void CipGraph::initialize() {
    if (_zeroLink) {
        _logger.debug("Graph initialize start");

        _zeroLink->graphInitialie(_handle, _config);
        _executor = _zeroLink->createExecutor(_handle, _config);

        _logger.debug("Graph initialize finish");
    }
}

CipGraph::~CipGraph() {
    if (_handle != nullptr) {
        auto result = _zeroLink->release(_handle);

        if (ZE_RESULT_SUCCESS == result) {
            _handle = nullptr;
        }
    }
}

}  // namespace intel_npu
