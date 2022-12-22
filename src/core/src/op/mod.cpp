// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/mod.hpp"

#include "itt.hpp"
#include "ngraph/runtime/reference/mod.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ v1 -------------------------------------------

op::v1::Mod::Mod(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::Mod::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Mod_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Mod>(new_args.at(0), new_args.at(1), this->get_autob());
}

namespace mod_op {
namespace {
template <typename T>
bool evaluate(const ov::Tensor& arg0,
              const ov::Tensor& arg1,
              const ov::Tensor& out,
              const op::AutoBroadcastSpec& broadcast_spec) {
    runtime::reference::mod(arg0.data<T>(),
                            arg1.data<T>(),
                            out.data<T>(),
                            arg0.get_shape(),
                            arg1.get_shape(),
                            broadcast_spec);
    return true;
}

bool evaluate_mod(const ov::Tensor& arg0,
                  const ov::Tensor& arg1,
                  const ov::Tensor& out,
                  const op::AutoBroadcastSpec& broadcast_spec) {
    bool rc = true;
    switch (arg0.get_element_type()) {
    case ov::element::Type_t::i8: {
        rc = evaluate<int8_t>(arg0, arg1, out, broadcast_spec);
    } break;
    case ov::element::Type_t::i16: {
        rc = evaluate<int16_t>(arg0, arg1, out, broadcast_spec);
    } break;
    case ov::element::Type_t::i32: {
        rc = evaluate<int32_t>(arg0, arg1, out, broadcast_spec);
    } break;
    case ov::element::Type_t::i64: {
        rc = evaluate<int64_t>(arg0, arg1, out, broadcast_spec);
    } break;
    case ov::element::Type_t::u8: {
        rc = evaluate<uint8_t>(arg0, arg1, out, broadcast_spec);
    } break;
    case ov::element::Type_t::u16: {
        rc = evaluate<uint16_t>(arg0, arg1, out, broadcast_spec);
    } break;
    case ov::element::Type_t::u32: {
        rc = evaluate<uint32_t>(arg0, arg1, out, broadcast_spec);
    } break;
    case ov::element::Type_t::u64: {
        rc = evaluate<uint64_t>(arg0, arg1, out, broadcast_spec);
    } break;
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace mod_op

bool op::v1::Mod::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Mod_evaluate);
    return mod_op::evaluate_mod(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v1::Mod::has_evaluate() const {
    OV_OP_SCOPE(v1_Mod_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
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
