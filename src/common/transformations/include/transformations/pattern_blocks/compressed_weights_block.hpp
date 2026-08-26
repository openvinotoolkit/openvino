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
    /// \param allow_two_level_scale  Also match a group scale that is itself quantized, i.e.
    ///     Constant(f8) -> Convert -> Multiply(Constant global) -> Convert, which is what NNCF emits
    ///     for NVFP4 and for INT2 experts built with --experts-fp8-scale. When it matches, the
    ///     "mul_const" anchor is the 8-bit group scale and the extra "scale_global_const" anchor
    ///     holds the per-tensor factor; a consumer that cannot apply that factor must leave this off,
    ///     or it will silently drop it and scale every weight wrong.
    CompressedWeightsBlock(const std::vector<ov::element::Type>& supported_weights_types,
                           const std::set<size_t>& supported_weights_ranks,
                           bool allow_two_level_scale = false);
};
