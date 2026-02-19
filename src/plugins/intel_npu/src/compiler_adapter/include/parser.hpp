// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "intel_npu/common/iparser.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace intel_npu {

class Parser final : public IParser {
public:
    Parser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct);

    std::shared_ptr<IGraph> parse(
        const ov::Tensor& mainBlob,
        const FilteredConfig& config,
        const std::optional<std::vector<ov::Tensor>>& initBlobs = std::nullopt,
        std::optional<std::shared_ptr<const ov::Model>>&& model = std::nullopt) const override;

private:
    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;
    std::shared_ptr<ZeGraphExtWrappers> _zeGraphExt;

    Logger _logger;
};

}  // namespace intel_npu
