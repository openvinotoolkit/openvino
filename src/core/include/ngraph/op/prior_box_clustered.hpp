// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/prior_box_clustered.hpp"

namespace ngraph {
namespace op {
using PriorBoxClusteredAttrs = ov::op::v0::PriorBoxClustered::Attributes;

namespace v0 {
using ov::op::v0::PriorBoxClustered;
}  // namespace v0
using v0::PriorBoxClustered;
}  // namespace op
}  // namespace ngraph
