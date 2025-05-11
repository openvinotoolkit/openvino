// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace pass {

// This transformation replaces a pattern containing complex numbers with simpler one:
// Original pattern:
//  Rfft2d->Reshape->ComplexAbs
//
// Replaced with:
//  RDFT -> Split -(real)-> Unsqueeze -> Reshape -> Square -> Add -> Sqrt
//               \-(imag)-> Unsqueeze -> Reshape -> Square /
class Rfft2dSimplifier : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::tensorflow_lite::pass::Rfft2dSimplifier");
    Rfft2dSimplifier();
};

}  // namespace pass
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
