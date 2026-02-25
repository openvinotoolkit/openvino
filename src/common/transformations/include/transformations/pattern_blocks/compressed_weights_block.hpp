// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "ov_ops/fully_connected.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass::pattern::op {

class TRANSFORMATIONS_API CompressedWeightsBlock;

}  // namespace ov::pass::pattern::op

class ov::pass::pattern::op::CompressedWeightsBlock : public ov::pass::pattern::op::Block {
public:
    CompressedWeightsBlock(const std::vector<ov::element::Type>& supported_weights_types,
                           const std::set<size_t>& supported_weights_ranks);
};
