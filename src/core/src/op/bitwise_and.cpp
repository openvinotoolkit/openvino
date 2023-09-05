// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/bitwise_and.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::v13::BitwiseAnd::BitwiseAnd(const Output<Node>& arg0,
                                const Output<Node>& arg1,
                                const AutoBroadcastSpec& auto_broadcast)
                                : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v13::BitwiseAnd::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v13_BitwiseAnd_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v13::BitwiseAnd>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool op::v13::BitwiseAnd::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v13_BitwiseAnd_evaluate);
    return true;
}

bool op::v13::BitwiseAnd::has_evaluate() const {
    OV_OP_SCOPE(v13_BitwiseAnd_has_evaluate);
    switch (get_input_element_type(0)) {
        case ngraph::element::i4:
        case ngraph::element::i8:
        case ngraph::element::i16:
        case ngraph::element::i32:
        case ngraph::element::i64:
        case ngraph::element::u4:
        case ngraph::element::u8:
        case ngraph::element::u16:
        case ngraph::element::u32:
        case ngraph::element::u64:
            return true;
        default:
            break;
    }
    return false;
}
