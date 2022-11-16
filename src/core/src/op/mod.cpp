// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/mod.hpp"

#include "itt.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ v1 -------------------------------------------

BWDCMP_RTTI_DEFINITION(op::v1::Mod);

op::v1::Mod::Mod(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::Mod::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Mod_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Mod>(new_args.at(0), new_args.at(1), this->get_autob());
}
