// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "driver_compiler_utils.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "izero_link.hpp"
#include "npu.hpp"
#include "zero_backend.hpp"

namespace intel_npu {

class DriverCompilerAdapter final : public ICompilerAdapter {
public:
    DriverCompilerAdapter(const std::shared_ptr<IEngineBackend>& iEngineBackend);

    std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    std::shared_ptr<IGraph> parse(const std::vector<uint8_t>& network, const Config& config) const override;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

private:
    std::shared_ptr<ZeroEngineBackend> _zeroBackend;
    std::shared_ptr<IZeroLink> _zeroLink;

    ze_device_graph_properties_t _deviceGraphProperties{};

    Logger _logger;
};

}  // namespace intel_npu
