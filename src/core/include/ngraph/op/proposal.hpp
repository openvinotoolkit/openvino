// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/proposal.hpp"

namespace ngraph {
namespace op {
using ProposalAttrs = ov::op::v1::Proposal::Attributes;

namespace v0 {
using ov::op::v1::Proposal;
}  // namespace v0

namespace v4 {
using ov::op::v4::Proposal;
}  // namespace v4
using v0::Proposal;
}  // namespace op
}  // namespace ngraph
