// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_adapter.hpp"

#include "compiler.hpp"
#include "driver_compiler_adapter.hpp"

namespace intel_npu {

std::vector<uint8_t>& CiDGraph::getCompiledNetwork() {
    auto cidCompiler = std::dynamic_pointer_cast<driverCompilerAdapter::LevelZeroCompilerAdapter>(_compiler._ptr);
    cidCompiler->getCompiledNetwork(_graphHandle, _compiledNetwork);
    return _compiledNetwork;
}

CompilerAdapter::CompilerAdapter(std::shared_ptr<NPUBackends> npuBackends, ov::intel_npu::CompilerType compilerType)
    : _compilerType(compilerType),
      _logger("CompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize CompilerAdapter start");
    _compiler = createCompiler(npuBackends, compilerType);
}

uint32_t CompilerAdapter::getSupportedOpsetVersion() const {
    return _compiler->getSupportedOpsetVersion();
}

std::shared_ptr<IGraph> CompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                 const Config& config) const {
    _logger.debug("compile start");
    switch (_compilerType) {
    case ov::intel_npu::CompilerType::MLIR:
        return std::make_shared<CiPGraph>(std::make_shared<NetworkDescription>(_compiler->compile(model, config)));
    case ov::intel_npu::CompilerType::DRIVER:
#ifdef ENABLE_DRIVER_COMPILER_ADAPTER
        auto cidCompiler = std::dynamic_pointer_cast<driverCompilerAdapter::LevelZeroCompilerAdapter>(_compiler._ptr);
        std::pair<NetworkDescription, void*> result = cidCompiler->compileAndReturnGraph(model, config);
        return std::make_shared<CiDGraph>(result.second, std::move(result.first.metadata));
#else
        OPENVINO_THROW("NPU Compiler Adapter is not enabled");
#endif
    }
    OPENVINO_THROW("Invalid NPU_COMPILER_TYPE");
}

ov::SupportedOpsMap CompilerAdapter::query(const std::shared_ptr<const ov::Model>& model, const Config& config) const {
    _logger.debug("query start");
    return _compiler->query(model, config);
}

std::shared_ptr<IGraph> CompilerAdapter::parse(std::vector<uint8_t>& network, const Config& config) const {
    _logger.debug("parse start");
    switch (_compilerType) {
    case ov::intel_npu::CompilerType::MLIR:
        return std::make_shared<CiPGraph>(
            std::make_shared<NetworkDescription>(std::move(network), _compiler->parse(network, config)));
    case ov::intel_npu::CompilerType::DRIVER:
#ifdef ENABLE_DRIVER_COMPILER_ADAPTER
        auto cidCompiler = std::dynamic_pointer_cast<driverCompilerAdapter::LevelZeroCompilerAdapter>(_compiler._ptr);
        std::pair<NetworkMetadata, void*> result = cidCompiler->parseAndReturnGraph(network, config);
        // There is an instance of the blob maintained inside the driver
        // and we can release the copy of the blob here to reduce memory consumption.
        network.clear();
        network.shrink_to_fit();
        return std::make_shared<CiDGraph>(result.second, result.first);
#else
        OPENVINO_THROW("NPU Compiler Adapter is not enabled");
#endif
    }
    OPENVINO_THROW("Invalid NPU_COMPILER_TYPE");
}

std::vector<ov::ProfilingInfo> CompilerAdapter::process_profiling_output(const std::vector<uint8_t>&,
                                                                         const std::vector<uint8_t>&,
                                                                         const Config&) const {
    OPENVINO_THROW("Profiling post-processing is not implemented.");
}

ov::SoPtr<ICompiler> CompilerAdapter::get_compiler() {
    return _compiler;
}

}  // namespace intel_npu
