// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/util/embeddingbag_offsets_base.hpp"
#include "ngraph/op/util/index_reduction.hpp"
#include "openvino/op/embeddingbag_offsets_sum.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::EmbeddingBagOffsetsSum;
}  // namespace v3
using v3::EmbeddingBagOffsetsSum;
}  // namespace op
}  // namespace ngraph
