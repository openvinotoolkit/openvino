// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/set_rows_op.hpp"

namespace ov {
namespace frontend {
namespace gguf {

SetRows::SetRows(const ov::Output<ov::Node>& data,
                 const ov::Output<ov::Node>& indices,
                 const ov::Output<ov::Node>& dst)
    : ov::op::Op({data, indices, dst}) {
    constructor_validate_and_infer_types();
}

void SetRows::validate_and_infer_types() {
    // The updated tensor is read like the destination: same element type and shape. Lowering
    // replaces this op before compile, so this only has to satisfy graph validation meanwhile.
    set_output_type(0, get_input_element_type(2), get_input_partial_shape(2));
}

std::shared_ptr<ov::Node> SetRows::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SetRows>(new_args[0], new_args[1], new_args[2]);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
