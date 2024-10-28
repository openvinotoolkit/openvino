// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_graph.hpp"

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/runtime.hpp"

namespace intel_npu {

PluginGraph::PluginGraph(const std::shared_ptr<IZeroAdapter>& adapter,
                         const ov::SoPtr<ICompiler>& compiler,
                         ze_graph_handle_t graphHandle,
                         NetworkMetadata metadata,
                         const std::vector<uint8_t> compiledNetwork,
                         const Config& config)
    : IGraph(graphHandle, std::move(metadata)),
      _adapter(adapter),
      _compiler(compiler),
      _compiledNetwork(std::move(compiledNetwork)),
      _logger("PluginGraph", config.get<LOG_LEVEL>()) {
    if (config.get<CREATE_EXECUTOR>()) {
        initialize(config);
    } else {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
    }
}

CompiledNetwork PluginGraph::export_blob() const {
    return CompiledNetwork(_compiledNetwork.data(), _compiledNetwork.size(), _compiledNetwork);
}

std::vector<ov::ProfilingInfo> PluginGraph::process_profiling_output(const std::vector<uint8_t>& profData,
                                                                     const Config& config) const {
    return _compiler->process_profiling_output(profData, _compiledNetwork, config);
}

void PluginGraph::set_argument_value(uint32_t argi, const void* argv) const {
    if (_adapter == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _adapter->setArgumentValue(_handle, argi, argv);
}

void PluginGraph::initialize(const Config& config) {
    if (_adapter) {
        _logger.debug("Graph initialize start");

        std::tie(_input_descriptors, _output_descriptors) = _adapter->getIODesc(_handle);
        _command_queue = _adapter->crateCommandQueue(config);

        if (config.has<WORKLOAD_TYPE>()) {
            set_workload_type(config.get<WORKLOAD_TYPE>());
        }

        _adapter->graphInitialize(_handle, config);

        _logger.debug("Graph initialize finish");
    }
}

PluginGraph::~PluginGraph() {
    if (_handle != nullptr) {
        auto result = _adapter->release(_handle);

        if (ZE_RESULT_SUCCESS == result) {
            _handle = nullptr;
        }
    }
}

}  // namespace intel_npu
