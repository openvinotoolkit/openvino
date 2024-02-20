// Copyright (C) 2018-2024 Intel Corporation
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

// This transformation handles GRUBlockCell with just one output - hidden state
class GRUBlockCellReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::GRUBlockCellReplacer");
    GRUBlockCellReplacer();
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
