// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/roll.hpp"

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v7::Roll>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    const auto& shiftType = inputs[1]->get_element_type();
    std::vector<int64_t> shift_int64;
    if (shiftType == ngraph::element::Type_t::i32) {
        auto shift = inputs[1]->get_data_ptr<const int32_t>();
        shift_int64.resize(ngraph::shape_size(inputs[1]->get_shape()));
        std::transform(shift,
                       shift + ngraph::shape_size(inputs[1]->get_shape()),
                       shift_int64.begin(),
                       [](const int32_t& elem) {
                           return static_cast<int64_t>(elem);
                       });
    }
    const auto& axesType = inputs[2]->get_element_type();
    std::vector<int64_t> axes_int64;
    if (axesType == ngraph::element::Type_t::i32) {
        auto axes = inputs[2]->get_data_ptr<const int32_t>();
        axes_int64.resize(ngraph::shape_size(inputs[2]->get_shape()));
        std::transform(axes,
                       axes + ngraph::shape_size(inputs[2]->get_shape()),
                       axes_int64.begin(),
                       [](const int32_t& elem) {
                           return static_cast<int64_t>(elem);
                       });
    }
    ngraph::runtime::reference::roll(
        inputs[0]->get_data_ptr<const char>(),
        inputs[1]->get_element_type() != ngraph::element::Type_t::i64 ? shift_int64.data()
                                                                      : inputs[1]->get_data_ptr<const int64_t>(),
        inputs[2]->get_element_type() != ngraph::element::Type_t::i64 ? axes_int64.data()
                                                                      : inputs[2]->get_data_ptr<const int64_t>(),
        outputs[0]->get_data_ptr<char>(),
        inputs[0]->get_shape(),
        inputs[1]->get_shape(),
        inputs[2]->get_shape(),
        inputs[0]->get_element_type().size());
    return true;
}