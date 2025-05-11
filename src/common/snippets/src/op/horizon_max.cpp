// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/op/horizon_max.hpp"

namespace ov {
namespace snippets {
namespace op {

HorizonMax::HorizonMax(const Output<Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> HorizonMax::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(HorizonMax_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<HorizonMax>(new_args.at(0));
}

void HorizonMax::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(HorizonMax_validate_and_infer_types);
    auto new_shape = get_input_partial_shape(0);
    if (!ov::is_scalar(new_shape)) {
        new_shape[new_shape.size() - 1] = 1lu;
    }
    set_output_type(0, get_input_element_type(0), new_shape);
}

} // namespace op
} // namespace snippets
} // namespace ov
