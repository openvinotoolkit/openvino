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

// Replace internal operation TensorListGetItem with a sub-graph that gets a tensor from container by index
class TensorListGetItemReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::TensorListGetItemReplacer");
    TensorListGetItemReplacer();
};

// Replace TensorListSetItem with concatenated output and TensorListGetItem with sliced input in Loop operation
class TensorListSliceInputAndConcatOutputReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::TensorListSliceInputAndConcatOutputReplacer");
    TensorListSliceInputAndConcatOutputReplacer();
};

// Replace and optimize sub-graphs with TensorList operations such as TensorListReserve,
// TensorListSetItem, TensorListGetItem
class TensorListOperationsResolver : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("TensorListOperationsResolver", "0");
    TensorListOperationsResolver() {
        add_matcher<TensorListReplacer>();
        add_matcher<TensorListSliceInputAndConcatOutputReplacer>();
        add_matcher<TensorListSetItemReplacer>();
        add_matcher<TensorListGetItemReplacer>();
    }
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
