// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backends.hpp"
#include "intel_npu/al/icompiler.hpp"

namespace intel_npu {

class CiPGraph : public IGraph {
public:
    CiPGraph(std::shared_ptr<NetworkDescription> network) : _network(network) {}
    NetworkMetadata& getMetadata() override {
        return _network->metadata;
    }
    std::vector<uint8_t>& getCompiledNetwork() override {
        return _network->compiledNetwork;
    }

private:
    std::shared_ptr<NetworkDescription> _network;
};

class CiDGraph : public IGraph {
public:
    CiDGraph(void* graphHandle, NetworkMetadata metadata) : _graphHandle(graphHandle), _metadata(std::move(metadata)) {}
    NetworkMetadata& getMetadata() override {
        return _metadata;
    }
    std::vector<uint8_t>& getCompiledNetwork() override;
    void releaseCompiledNetwork() {
        _compiledNetwork.clear();
        _compiledNetwork.shrink_to_fit();
    }
    void* getGraphHandle() {
        return _graphHandle;
    }

private:
    void* _graphHandle = nullptr;
    NetworkMetadata _metadata;
    std::vector<uint8_t> _compiledNetwork;
    ov::SoPtr<ICompiler> _compiler;
};

class CompilerAdapter {
public:
    CompilerAdapter(std::shared_ptr<NPUBackends> npuBackends, ov::intel_npu::CompilerType compilerType);

    uint32_t getSupportedOpsetVersion() const;

    std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const;

    std::shared_ptr<IGraph> parse(std::vector<uint8_t>& network, const Config& config) const;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const Config& config) const;
    ov::SoPtr<ICompiler> get_compiler();

private:
    ov::intel_npu::CompilerType _compilerType;
    Logger _logger;
    /**
     * @brief Separate externals calls to separate class
     */
    ov::SoPtr<ICompiler> _compiler;
};

}  //  namespace intel_npu