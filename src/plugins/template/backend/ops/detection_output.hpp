// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/detection_output.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::DetectionOutput>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::referenceDetectionOutput<T> refDetOut(op->get_attrs(),
                                                                      op->get_input_shape(0),
                                                                      op->get_input_shape(1),
                                                                      op->get_input_shape(2),
                                                                      op->get_output_shape(0));
    if (op->get_input_size() == 3) {
        refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                      inputs[1]->get_data_ptr<const T>(),
                      inputs[2]->get_data_ptr<const T>(),
                      nullptr,
                      nullptr,
                      outputs[0]->get_data_ptr<T>());
    } else if (op->get_input_size() == 5) {
        refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                      inputs[1]->get_data_ptr<const T>(),
                      inputs[2]->get_data_ptr<const T>(),
                      inputs[3]->get_data_ptr<const T>(),
                      inputs[4]->get_data_ptr<const T>(),
                      outputs[0]->get_data_ptr<T>());
    } else {
        throw ngraph::ngraph_error("DetectionOutput layer supports only 3 or 5 inputs");
    }
    return true;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v8::DetectionOutput>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::referenceDetectionOutput<T> refDetOut(op->get_attrs(),
                                                                      op->get_input_shape(0),
                                                                      op->get_input_shape(1),
                                                                      op->get_input_shape(2),
                                                                      op->get_output_shape(0));
    if (op->get_input_size() == 3) {
        refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                      inputs[1]->get_data_ptr<const T>(),
                      inputs[2]->get_data_ptr<const T>(),
                      nullptr,
                      nullptr,
                      outputs[0]->get_data_ptr<T>());
    } else if (op->get_input_size() == 5) {
        refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                      inputs[1]->get_data_ptr<const T>(),
                      inputs[2]->get_data_ptr<const T>(),
                      inputs[3]->get_data_ptr<const T>(),
                      inputs[4]->get_data_ptr<const T>(),
                      outputs[0]->get_data_ptr<T>());
    } else {
        throw ngraph::ngraph_error("DetectionOutput layer supports only 3 or 5 inputs");
    }
    return true;
}