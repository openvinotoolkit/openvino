// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

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
