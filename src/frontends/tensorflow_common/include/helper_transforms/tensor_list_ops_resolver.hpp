// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

// Replace internal operation TensorListReserve with a sub-graph producing initial container
class TensorListReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::TensorListReplacer");
    TensorListReplacer();
};

// Replace internal operation TensorListSetItem with a sub-graph that inserts a new tensor into container
class TensorListSetItemReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::TensorListSetItemReplacer");
    TensorListSetItemReplacer();
};

// Replace internal operation TensorListPushBack with a sub-graph
// that inserts a new tensor into the tail of the container
class TensorListPushBackReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::TensorListPushBackReplacer");
    TensorListPushBackReplacer();
};

// Replace internal operation TensorListGetItem with a sub-graph that gets a tensor from container by index
class TensorListGetItemReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::TensorListGetItemReplacer");
    TensorListGetItemReplacer();
};

// Optimize sub-graphs with TensorList operations in Loop body graph
// Replace TensorListSetItem and TensorListGetItem with ConcatOutput and SlicedInput
class TensorListInLoopOptimization : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::TensorListInLoopOptimization");
    TensorListInLoopOptimization();
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
