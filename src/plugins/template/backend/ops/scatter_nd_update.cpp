// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/scatter_nd_update.hpp"

#include "evaluates_map.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v3::ScatterNDUpdate>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    auto idxType = op->get_input_element_type(1);
    if (idxType == ov::element::i32) {
        ngraph::runtime::reference::scatterNdUpdate<T, int32_t>(inputs[0]->get_data_ptr<const T>(),
                                                                inputs[1]->get_data_ptr<const int32_t>(),
                                                                inputs[2]->get_data_ptr<const T>(),
                                                                outputs[0]->get_data_ptr<T>(),
                                                                op->get_input_shape(0),
                                                                op->get_input_shape(1),
                                                                op->get_input_shape(2));
    } else if (idxType == ov::element::i64) {
        ngraph::runtime::reference::scatterNdUpdate<T, int64_t>(inputs[0]->get_data_ptr<const T>(),
                                                                inputs[1]->get_data_ptr<const int64_t>(),
                                                                inputs[2]->get_data_ptr<const T>(),
                                                                outputs[0]->get_data_ptr<T>(),
                                                                op->get_input_shape(0),
                                                                op->get_input_shape(1),
                                                                op->get_input_shape(2));
    } else {
        throw ngraph::ngraph_error("ScatterNDUpdate layer support only i32 and i64 'indices' input precision!");
    }
    return true;
}