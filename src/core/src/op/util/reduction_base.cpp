// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/reduction_base.hpp"

#include "reduce_shape_inference.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::util::ReductionBase);

ov::op::util::ReductionBase::ReductionBase() = default;

ov::op::util::ReductionBase::ReductionBase(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : Op({arg, reduction_axes}) {}

ov::PartialShape ov::op::util::ReductionBase::infer_reduction_output_shape(const bool keep_dims) {
    ov::PartialShape output_shape;
    reduce_shape_infer(this, keep_dims, get_input_partial_shape(0), output_shape);
    return output_shape;
}
