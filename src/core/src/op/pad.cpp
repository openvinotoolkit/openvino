// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pad.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "pad_shape_inference.hpp"

namespace ov {

op::v1::Pad::Pad(const Output<Node>& arg,
                 const Output<Node>& pads_begin,
                 const Output<Node>& pads_end,
                 const Output<Node>& arg_pad_value,
                 op::PadMode pad_mode)
    : op::util::PadBase(arg, pads_begin, pads_end, arg_pad_value, pad_mode) {
    constructor_validate_and_infer_types();
}

op::v1::Pad::Pad(const Output<Node>& arg,
                 const Output<Node>& pads_begin,
                 const Output<Node>& pads_end,
                 op::PadMode pad_mode)
    : op::util::PadBase(arg,
                        pads_begin,
                        pads_end,
                        op::v0::Constant::create(arg.get_element_type(), Shape{}, {0}),
                        pad_mode) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v1::Pad::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Pad_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (get_input_size() == 4) {
        return std::make_shared<op::v1::Pad>(new_args.at(0),
                                             new_args.at(1),
                                             new_args.at(2),
                                             new_args.at(3),
                                             m_pad_mode);
    } else {
        return std::make_shared<op::v1::Pad>(new_args.at(0), new_args.at(1), new_args.at(2), m_pad_mode);
    }
}

bool op::v1::Pad::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Pad_evaluate);
    return evaluate_pad(outputs, inputs);
}

bool op::v1::Pad::has_evaluate() const {
    OV_OP_SCOPE(v1_Pad_has_evaluate);
    return true;
}

op::v12::Pad::Pad(const Output<Node>& arg,
                  const Output<Node>& pads_begin,
                  const Output<Node>& pads_end,
                  const Output<Node>& arg_pad_value,
                  op::PadMode pad_mode)
    : op::util::PadBase(arg, pads_begin, pads_end, arg_pad_value, pad_mode) {
    constructor_validate_and_infer_types();
}

op::v12::Pad::Pad(const Output<Node>& arg,
                  const Output<Node>& pads_begin,
                  const Output<Node>& pads_end,
                  op::PadMode pad_mode)
    : op::util::PadBase(arg,
                        pads_begin,
                        pads_end,
                        op::v0::Constant::create(arg.get_element_type(), ov::Shape{}, {0}),
                        pad_mode) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v12::Pad::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v12_Pad_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (get_input_size() == 4) {
        return std::make_shared<op::v12::Pad>(new_args.at(0),
                                              new_args.at(1),
                                              new_args.at(2),
                                              new_args.at(3),
                                              m_pad_mode);
    } else {
        return std::make_shared<op::v12::Pad>(new_args.at(0), new_args.at(1), new_args.at(2), m_pad_mode);
    }
}

bool op::v12::Pad::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v12_Pad_evaluate);
    return evaluate_pad(outputs, inputs);
}

bool op::v12::Pad::has_evaluate() const {
    OV_OP_SCOPE(v12_Pad_has_evaluate);
    return true;
}

}  // namespace ov
