// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fft_base.hpp"
#include "openvino/op/idft.hpp"

namespace ngraph {
namespace op {
namespace v7 {
using ov::op::v7::IDFT;
}  // namespace v7
}  // namespace op
}  // namespace ngraph
