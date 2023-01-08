// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/type.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ngraph {
namespace op {
using ov::op::AutoBroadcastSpec;
using ov::op::AutoBroadcastType;
using ov::op::BroadcastModeSpec;
using ov::op::BroadcastType;
using ov::op::EpsMode;
using ov::op::PadMode;
using ov::op::PadType;
using ov::op::RecurrentSequenceDirection;
using ov::op::RoundingType;
using ov::op::TopKMode;
using ov::op::TopKSortType;
}  // namespace op
}  // namespace ngraph
