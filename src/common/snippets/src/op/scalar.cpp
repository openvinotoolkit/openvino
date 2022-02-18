// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/scalar.hpp"

using namespace ngraph;

std::shared_ptr<Node> snippets::op::Scalar::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Scalar>(*this);
}

// Scalar currently supports only one-element constants, this could be changed in the future
void snippets::op::Scalar::validate_and_infer_types() {
    Constant::validate_and_infer_types();
    auto out_pshape = get_output_partial_shape(0);
    NODE_VALIDATION_CHECK(this, out_pshape.is_static(), "Scalar supports only static input shapes");
    NODE_VALIDATION_CHECK(this, out_pshape.get_shape().empty() || ov::shape_size(out_pshape.get_shape()) == 1,
                      "Scalar supports only one-element constants, got ", out_pshape.get_shape(),
                      " shape");
}