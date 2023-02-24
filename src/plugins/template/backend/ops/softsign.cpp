// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/softsign.hpp"

#include "evaluates_map.hpp"
#include "openvino/op/softsign.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v9::SoftSign>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    ov::element::Type input_et = op->get_input_element_type(0);
    switch (input_et) {
    case ov::element::Type_t::f64:
        ngraph::runtime::reference::softsign<double>(inputs[0]->get_data_ptr<double>(),
                                                     outputs[0]->get_data_ptr<double>(),
                                                     ov::shape_size(inputs[0]->get_shape()));
        break;
    case ov::element::Type_t::f32:
        ngraph::runtime::reference::softsign<float>(inputs[0]->get_data_ptr<float>(),
                                                    outputs[0]->get_data_ptr<float>(),
                                                    ov::shape_size(inputs[0]->get_shape()));
        break;
    case ov::element::Type_t::f16:
        ngraph::runtime::reference::softsign<ov::float16>(inputs[0]->get_data_ptr<ov::float16>(),
                                                          outputs[0]->get_data_ptr<ov::float16>(),
                                                          ov::shape_size(inputs[0]->get_shape()));
        break;
    case ov::element::Type_t::bf16:
        ngraph::runtime::reference::softsign<ov::bfloat16>(inputs[0]->get_data_ptr<ov::bfloat16>(),
                                                           outputs[0]->get_data_ptr<ov::bfloat16>(),
                                                           ov::shape_size(inputs[0]->get_shape()));
        break;
    default:
        return false;
    }
    return true;
}
