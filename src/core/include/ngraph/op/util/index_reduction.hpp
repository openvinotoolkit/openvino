// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "ngraph/op/op.hpp"
#include "openvino/op/util/index_reduction.hpp"

namespace ngraph {
namespace op {
namespace util {
using ov::op::util::IndexReduction;
}  // namespace util
}  // namespace op
}  // namespace ngraph
