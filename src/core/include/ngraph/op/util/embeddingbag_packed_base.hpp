// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/util/index_reduction.hpp"
#include "openvino/op/util/embeddingbag_packed_base.hpp"

namespace ngraph {
namespace op {
namespace util {
using ov::op::util::EmbeddingBagPackedBase;
}  // namespace util
using util::EmbeddingBagPackedBase;
}  // namespace op
}  // namespace ngraph
