// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

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
