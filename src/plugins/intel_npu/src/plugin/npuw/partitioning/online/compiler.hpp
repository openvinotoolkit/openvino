// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "../partitioning.hpp"  // ov::npuw::Ensemble
#include "intel_npu/config/config.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace npuw {
namespace online {

ov::npuw::Ensemble buildPartitioning(const std::shared_ptr<ov::Model>& model,
                                     ::intel_npu::Config& cfg,
                                     const ov::npuw::v1::subgraphs::PatternRegistry* subgraph_patterns = nullptr);

}  // namespace online
}  // namespace npuw
}  // namespace ov
