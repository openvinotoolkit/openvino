// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "intel_npu/common/iparser.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/core/model.hpp"

namespace intel_npu {

class DynamicParser final : public IParser {
public:
    DynamicParser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct);

    std::shared_ptr<IGraph> parse(
        const ov::Tensor& mainBlob,
        const FilteredConfig& config,
        const std::optional<std::vector<ov::Tensor>>& initBlobs = std::nullopt,
        std::optional<std::shared_ptr<const ov::Model>>&& model = std::nullopt) const override;

private:
    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    Logger _logger;
};

}  // namespace intel_npu
