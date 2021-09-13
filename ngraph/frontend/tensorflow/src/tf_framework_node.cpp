// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tf_framework_node.hpp>

namespace ngraph {
namespace frontend {
NGRAPH_RTTI_DEFINITION(TFFrameworkNode, "TFFrameworkNode", 1);

void TFFrameworkNode::validate_and_infer_types() {
    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(i, ov::element::dynamic, PartialShape::dynamic());
    }
}

}  // namespace frontend
}  // namespace ngraph
