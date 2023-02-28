// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "openvino/op/convert_like.hpp"

namespace convert_like_v1 {
template <ov::element::Type_t ti, ov::element::Type_t to>
inline void evaluate(const std::shared_ptr<ov::op::v1::ConvertLike>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    outputs[0]->set_shape(inputs[0]->get_shape());
    size_t element_count = shape_size(outputs[0]->get_shape());

    if (((ti == ov::element::u1) || (to == ov::element::u1)) || ((ti == ov::element::u4) || (to == ov::element::u4)) ||
        ((ti == ov::element::i4) || (to == ov::element::i4))) {
        ngraph::runtime::reference::detail::lp_convert(inputs[0]->get_data_ptr<ti>(),
                                                       outputs[0]->get_data_ptr<to>(),
                                                       element_count,
                                                       ti,
                                                       to);
    } else {
        ngraph::runtime::reference::convert(inputs[0]->get_data_ptr<ti>(),
                                            outputs[0]->get_data_ptr<to>(),
                                            element_count);
    }
}
}  // namespace convert_like_v1

template <ov::element::Type_t OUT_ET>
bool evaluate(const std::shared_ptr<ov::op::v1::ConvertLike>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    switch (inputs[0]->get_element_type()) {
    case ov::element::Type_t::boolean:
        convert_like_v1::evaluate<ov::element::Type_t::boolean, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::u1:
        convert_like_v1::evaluate<ov::element::Type_t::u1, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::u4:
        convert_like_v1::evaluate<ov::element::Type_t::u4, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::u8:
        convert_like_v1::evaluate<ov::element::Type_t::u8, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::u16:
        convert_like_v1::evaluate<ov::element::Type_t::u16, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::u32:
        convert_like_v1::evaluate<ov::element::Type_t::u32, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::u64:
        convert_like_v1::evaluate<ov::element::Type_t::u64, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i4:
        convert_like_v1::evaluate<ov::element::Type_t::i4, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i8:
        convert_like_v1::evaluate<ov::element::Type_t::i8, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i16:
        convert_like_v1::evaluate<ov::element::Type_t::i16, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i32:
        convert_like_v1::evaluate<ov::element::Type_t::i32, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i64:
        convert_like_v1::evaluate<ov::element::Type_t::i64, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::bf16:
        convert_like_v1::evaluate<ov::element::Type_t::bf16, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::f16:
        convert_like_v1::evaluate<ov::element::Type_t::f16, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::f32:
        convert_like_v1::evaluate<ov::element::Type_t::f32, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::Type_t::f64:
        convert_like_v1::evaluate<ov::element::Type_t::f64, OUT_ET>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}
