// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/frontend/tensorflow/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

// This transformation handles GRUBlockCell with just one output - hidden state
class TENSORFLOW_API GRUBlockCellReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::GRUBlockCellReplacer");
    GRUBlockCellReplacer();
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
