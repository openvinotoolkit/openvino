// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/interpolate.hpp"

#include "evaluates_map.hpp"
#include "openvino/op/interpolate.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::Interpolate>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    ov::element::Type input_et = op->get_input_element_type(0);
    switch (input_et) {
    case ov::element::Type_t::f64:
        ngraph::runtime::reference::interpolate<double>(inputs[0]->get_data_ptr<double>(),
                                                        op->get_input_partial_shape(0),
                                                        outputs[0]->get_data_ptr<double>(),
                                                        op->get_output_shape(0),
                                                        op->get_attrs());
        break;
    case ov::element::Type_t::f32:
        ngraph::runtime::reference::interpolate<float>(inputs[0]->get_data_ptr<float>(),
                                                       op->get_input_partial_shape(0),
                                                       outputs[0]->get_data_ptr<float>(),
                                                       op->get_output_shape(0),
                                                       op->get_attrs());
        break;
    case ov::element::Type_t::f16:
        ngraph::runtime::reference::interpolate<ov::float16>(inputs[0]->get_data_ptr<ov::float16>(),
                                                             op->get_input_partial_shape(0),
                                                             outputs[0]->get_data_ptr<ov::float16>(),
                                                             op->get_output_shape(0),
                                                             op->get_attrs());
        break;
    case ov::element::Type_t::bf16:
        ngraph::runtime::reference::interpolate<ov::bfloat16>(inputs[0]->get_data_ptr<ov::bfloat16>(),
                                                              op->get_input_partial_shape(0),
                                                              outputs[0]->get_data_ptr<ov::bfloat16>(),
                                                              op->get_output_shape(0),
                                                              op->get_attrs());
        break;
    default:;
    }
    return true;
}
