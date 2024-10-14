// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "icompiler_adapter.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

class CipCompilerAdapter final : public ICompilerAdapter {
public:
    CipCompilerAdapter(const std::shared_ptr<IEngineBackend>& iEngineBackend);

    std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    std::shared_ptr<IGraph> parse(const std::vector<uint8_t>& network, const Config& config) const override;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

private:
    std::shared_ptr<IEngineBackend> _iEngineBackend;
    ov::SoPtr<ICompiler> _compiler;

    Logger _logger;
};

}  // namespace intel_npu
