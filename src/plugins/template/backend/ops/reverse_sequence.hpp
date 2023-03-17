// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"

namespace reverse_sequence_v0 {
template <ngraph::element::Type_t t1, ngraph::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ngraph::op::v0::ReverseSequence>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<t1>::value_type;
    using T2 = typename ngraph::element_type_traits<t2>::value_type;
    ngraph::runtime::reference::reverse_sequence<T1, T2>(inputs[0]->get_data_ptr<T1>(),
                                                         outputs[0]->get_data_ptr<T1>(),
                                                         inputs[0]->get_shape(),
                                                         op->get_batch_axis(),
                                                         op->get_sequence_axis(),
                                                         inputs[1]->get_data_ptr<T2>());
}
}  // namespace reverse_sequence_v0

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::ReverseSequence>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[1]->get_element_type()) {
    case ngraph::element::Type_t::boolean:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::boolean>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i8:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::i8>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i16:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::i16>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i32:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::i32>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i64:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::u8:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::u8>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::u16:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::u16>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::u32:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::u32>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::u64:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::u64>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::f16:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::f16>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::f32:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::f32>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::f64:
        reverse_sequence_v0::evaluate<ET, ngraph::element::Type_t::f64>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}