// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cid_graph.hpp"

namespace intel_npu {

CidGraph::CidGraph(const std::shared_ptr<ILevelZeroCompilerInDriver>& apiAdapter,
                   ze_graph_handle_t graphHandle,
                   NetworkMetadata metadata,
                   const Config& config)
    : IGraph(static_cast<void*>(graphHandle), std::move(metadata)),
      _apiAdapter(apiAdapter),
      _config(config),
      _logger("CidGraph", _config.get<LOG_LEVEL>()) {
    initialize();
}

CompiledNetwork CidGraph::export_blob() const {
    return _apiAdapter->getCompiledNetwork(static_cast<ze_graph_handle_t>(_handle));
}

std::vector<ov::ProfilingInfo> CidGraph::process_profiling_output() const {
    OPENVINO_THROW("Profiling post-processing is not implemented.");
}

void CidGraph::set_argument_value(uint32_t argi, const void* argv) const {
    _apiAdapter->setArgumentValue(static_cast<ze_graph_handle_t>(_handle), argi, argv);
}

void CidGraph::initialize() const {
    _apiAdapter->graphInitialie(static_cast<ze_graph_handle_t>(_handle), _config);
}

CidGraph::~CidGraph() {
    if (_handle != nullptr) {
        auto result = _apiAdapter->release(static_cast<ze_graph_handle_t>(_handle));

        if (ZE_RESULT_SUCCESS == result) {
            _handle = nullptr;
        }
    }
}

}  // namespace intel_npu
