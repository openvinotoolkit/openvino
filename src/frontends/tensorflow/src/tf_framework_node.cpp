// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tf_framework_node.hpp>

namespace ov {
namespace frontend {
namespace tensorflow {

void FrameworkNode::validate_and_infer_types() {
    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(i, ov::element::dynamic, PartialShape::dynamic());
    }
}
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
