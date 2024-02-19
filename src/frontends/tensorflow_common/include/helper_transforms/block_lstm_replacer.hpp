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

// This transformation replaces BlockLSTM with such outputs as concatenated hidden states
// and cell state from the last time step.
class BlockLSTMReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::BlockLSTMReplacer");
    BlockLSTMReplacer();
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
