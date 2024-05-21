// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "openvino/openvino.hpp"
#include "../partitioning.hpp" // ov::npuw::Ensemble

namespace ov {
namespace npuw {
namespace online {

ov::npuw::Ensemble buildPartitioning(const std::shared_ptr<ov::Model>& model);

} // namespace online
} // namespace npuw
} // namespace ov
