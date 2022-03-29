// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/generate_proposals.hpp"

namespace ngraph {
namespace op {
namespace v9 {
using ov::op::v9::GenerateProposals;
}  // namespace v9
}  // namespace op
}  // namespace ngraph
