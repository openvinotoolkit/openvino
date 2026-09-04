// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/horizon.hpp"

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/op.hpp"
#include "snippets/itt.hpp"

namespace ov::snippets::op {

HorizonBase::HorizonBase(const Output<Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}

HorizonBase::HorizonBase(const OutputVector& x) : Op(x) {
    constructor_validate_and_infer_types();
}

void HorizonBase::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(HorizonBase_validate_and_infer_types);
    auto new_shape = get_input_partial_shape(0);
    if (!ov::is_scalar(new_shape)) {
        new_shape[new_shape.size() - 1] = 1LU;
    }
    set_output_type(0, get_input_element_type(0), new_shape);
}

}  // namespace ov::snippets::op
