// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/is_nan.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v10::IsNaN>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    ngraph::element::Type input_et = op->get_input_element_type(0);
    switch (input_et) {
    case ngraph::element::Type_t::f64:
        ngraph::runtime::reference::is_nan(inputs[0]->get_data_ptr<double>(),
                                           outputs[0]->get_data_ptr<ngraph::element::Type_t::boolean>(),
                                           ngraph::shape_size(inputs[0]->get_shape()));
        break;
    case ngraph::element::Type_t::f32:
        ngraph::runtime::reference::is_nan(inputs[0]->get_data_ptr<float>(),
                                           outputs[0]->get_data_ptr<ngraph::element::Type_t::boolean>(),
                                           ngraph::shape_size(inputs[0]->get_shape()));
        break;
    case ngraph::element::Type_t::f16:
        ngraph::runtime::reference::is_nan(inputs[0]->get_data_ptr<ngraph::float16>(),
                                           outputs[0]->get_data_ptr<ngraph::element::Type_t::boolean>(),
                                           ngraph::shape_size(inputs[0]->get_shape()));
        break;
    case ngraph::element::Type_t::bf16:
        ngraph::runtime::reference::is_nan(inputs[0]->get_data_ptr<ngraph::bfloat16>(),
                                           outputs[0]->get_data_ptr<ngraph::element::Type_t::boolean>(),
                                           ngraph::shape_size(inputs[0]->get_shape()));
        break;
    default:
        return false;
    }
    return true;
}