// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/util/embeddingbag_packed_base.hpp"
#include "ngraph/op/util/index_reduction.hpp"
#include "openvino/op/embeddingbag_packedsum.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::EmbeddingBagPackedSum;
}  // namespace v3
using v3::EmbeddingBagPackedSum;
}  // namespace op
}  // namespace ngraph
