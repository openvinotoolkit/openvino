// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"

namespace intel_npu {

class ICompilerAdapter {
public:
    virtual std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model,
                                            const Config& config) const = 0;
    virtual std::shared_ptr<IGraph> parse(std::vector<uint8_t> network, const Config& config) const = 0;
    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const = 0;

    virtual ~ICompilerAdapter() = default;
};

}  // namespace intel_npu
