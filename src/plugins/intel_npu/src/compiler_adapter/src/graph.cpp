// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph.hpp"

#include "intel_npu/al/config/runtime.hpp"

namespace intel_npu {

Graph::Graph(const std::shared_ptr<IZeroLink>& zeroCompilerInDriver,
             ze_graph_handle_t graphHandle,
             NetworkMetadata& metadata,
             const Config& config)
    : IGraph(graphHandle, metadata),
      _zeroCompilerAdapter(zeroCompilerInDriver),
      _config(config),
      _logger("Graph", _config.get<LOG_LEVEL>()) {
    if (_config.get<CREATE_EXECUTOR>()) {
        initialize();
    } else {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
    }
}

Graph::Graph(const std::shared_ptr<IZeroLink>& zeroCompilerInDriver,
             const ov::SoPtr<ICompiler>& compiler,
             ze_graph_handle_t graphHandle,
             NetworkMetadata& metadata,
             const std::vector<uint8_t>& compiledNetwork,
             const Config& config)
    : IGraph(graphHandle, metadata),
      _zeroCompilerAdapter(zeroCompilerInDriver),
      _compiler(compiler),
      _compiledNetwork(compiledNetwork),
      _config(config),
      _logger("Graph", _config.get<LOG_LEVEL>()) {
    if (_config.get<CREATE_EXECUTOR>()) {
        initialize();
    } else {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
    }
}

CompiledNetwork Graph::export_blob() const {
    if (_compiler) {
        return CompiledNetwork(_compiledNetwork.data(), _compiledNetwork.size(), _compiledNetwork);
    }
    return _zeroCompilerAdapter->getCompiledNetwork(_handle);
}

std::vector<ov::ProfilingInfo> Graph::process_profiling_output(const std::vector<uint8_t>& profData) const {
    if (_compiler) {
        return _compiler->process_profiling_output(profData, _compiledNetwork, _config);
    }

    OPENVINO_THROW("Profiling post-processing is not supported.");
}

void Graph::set_argument_value(uint32_t argi, const void* argv) const {
    if (_zeroCompilerAdapter == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _zeroCompilerAdapter->setArgumentValue(_handle, argi, argv);
}

void Graph::initialize() {
    if (_zeroCompilerAdapter) {
        _logger.debug("Graph initialize start");

        _zeroCompilerAdapter->graphInitialie(_handle, _config);
        _executor = _zeroCompilerAdapter->createExecutor(_handle, _config);

        _logger.debug("Graph initialize finish");
    }
}

Graph::~Graph() {
    if (_handle != nullptr) {
        auto result = _zeroCompilerAdapter->release(_handle);

        if (ZE_RESULT_SUCCESS == result) {
            _handle = nullptr;
        }
    }
}

}  // namespace intel_npu
