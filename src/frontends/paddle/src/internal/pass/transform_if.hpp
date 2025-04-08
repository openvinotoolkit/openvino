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

class TransformIf : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::paddle::pass::TransformIf");
    TransformIf(std::vector<std::shared_ptr<Model>> functions);

private:
    std::vector<std::shared_ptr<Model>> m_functions;
};

}  // namespace pass
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
