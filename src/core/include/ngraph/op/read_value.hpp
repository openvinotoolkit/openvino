// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/op/util/variable_extension.hpp"
#include "openvino/op/read_value.hpp"

namespace ngraph {
namespace op {
using ov::op::util::ReadValueBase;

namespace v3 {
using ov::op::v3::ReadValue;
}  // namespace v3

namespace v6 {
using ov::op::v6::ReadValue;
}  // namespace v6
}  // namespace op
}  // namespace ngraph
