// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/bitwise_xor.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::v13::BitwiseXor::BitwiseXor(const Output<Node>& arg0,
                                const Output<Node>& arg1,
                                const AutoBroadcastSpec& auto_broadcast)
                                : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v13::BitwiseXor::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v13_BitwiseXor_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v13::BitwiseXor>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool op::v13::BitwiseXor::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v13_BitwiseXor_evaluate);
    return true;
}

bool op::v13::BitwiseXor::has_evaluate() const {
    OV_OP_SCOPE(v13_BitwiseXor_has_evaluate);
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
