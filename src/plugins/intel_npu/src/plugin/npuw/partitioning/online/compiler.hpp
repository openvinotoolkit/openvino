// Copyright (C) 2024 Intel Corporation
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

ov::npuw::Ensemble buildPartitioning(const std::shared_ptr<ov::Model>& model, ::intel_npu::Config& cfg);

}  // namespace online
}  // namespace npuw
}  // namespace ov
