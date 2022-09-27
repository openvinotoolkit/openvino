// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/squared_difference.hpp"

#include "itt.hpp"

using namespace std;

// ------------------------------ v0 -------------------------------------------

BWDCMP_RTTI_DEFINITION(ov::op::v0::SquaredDifference);

ov::op::v0::SquaredDifference::SquaredDifference(const Output<Node>& arg0,
                                                 const Output<Node>& arg1,
                                                 const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<ov::Node> ov::op::v0::SquaredDifference::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_SquaredDifference_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ov::op::v0::SquaredDifference>(new_args.at(0), new_args.at(1), this->get_autob());
}
