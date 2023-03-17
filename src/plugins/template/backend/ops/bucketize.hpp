// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/bucketize.hpp"

namespace bucketize_v3 {
template <ngraph::element::Type_t t1, ngraph::element::Type_t t2, ngraph::element::Type_t t3>
inline void evaluate(const std::shared_ptr<ngraph::op::v3::Bucketize>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<t1>::value_type;
    using T2 = typename ngraph::element_type_traits<t2>::value_type;
    using T3 = typename ngraph::element_type_traits<t3>::value_type;

    ngraph::runtime::reference::bucketize<T1, T2, T3>(inputs[0]->get_data_ptr<T1>(),
                                                      inputs[1]->get_data_ptr<T2>(),
                                                      outputs[0]->get_data_ptr<T3>(),
                                                      op->get_input_shape(0),
                                                      op->get_input_shape(1),
                                                      op->get_with_right_bound());
}

static inline constexpr uint16_t getElementMask(ngraph::element::Type_t type1, ngraph::element::Type_t type2) {
    return (static_cast<uint8_t>(type1)) | (static_cast<uint8_t>(type2) << 8);
}

}  // namespace bucketize_v3

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v3::Bucketize>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (bucketize_v3::getElementMask(op->get_input_element_type(0), op->get_input_element_type(1))) {
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f32, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::f16, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i32, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i32, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i64, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i64, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::i8, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::i8, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::f32):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::f32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::f16):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::f16, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::i32):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::i32, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::i64):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::i64, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::i8):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::i8, ET>(op, outputs, inputs);
        break;
    case bucketize_v3::getElementMask(ngraph::element::Type_t::u8, ngraph::element::Type_t::u8):
        bucketize_v3::evaluate<ngraph::element::Type_t::u8, ngraph::element::Type_t::u8, ET>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}