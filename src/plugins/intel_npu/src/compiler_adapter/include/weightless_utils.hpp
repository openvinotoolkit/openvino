// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/network_metadata.hpp"
#include "openvino/op/constant.hpp"

namespace intel_npu {

bool isInitMetadata(const NetworkMetadata& networkMetadata);

std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> get_all_constants_in_topological_order(
    const std::shared_ptr<const ov::Model>& model);

std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> get_all_constants_memory_mapped(
    const std::string& weightsPath,
    const std::vector<NetworkMetadata>& initNetworkMetadata);

}  // namespace intel_npu
