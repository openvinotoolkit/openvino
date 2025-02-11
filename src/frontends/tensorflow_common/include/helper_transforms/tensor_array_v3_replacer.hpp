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
namespace tensorflow {
namespace pass {

// This transformation replaces internal operation TensorArrayV3 with a Constant
// that simulates initial state of tensor array container
class TensorArrayV3Replacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::tensorflow::pass::TensorArrayV3Replacer");
    TensorArrayV3Replacer();
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
