// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace pass {

class TransformFakeQuantize : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::paddle::pass::TransformFakeQuantize");
    TransformFakeQuantize();

private:
};

}  // namespace pass
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
