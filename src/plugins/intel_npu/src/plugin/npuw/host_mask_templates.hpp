// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "openvino/core/node_output.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"

namespace ov::npuw::detail {

bool match_host_mask_template(const ov::Output<ov::Node>& mask,
                              const std::shared_ptr<ov::op::v0::Parameter>& row,
                              int64_t kind,
                              const ov::PartialShape& query_shape,
                              int64_t& f2_window);

}  // namespace ov::npuw::detail
