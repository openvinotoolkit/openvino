// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace test {

inline void make_u3_symmetric_zero_points(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(node);
        if (!subtract) {
            continue;
        }
        const auto weights_convert = ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(0));
        if (!weights_convert) {
            continue;
        }
        const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(weights_convert->get_input_node_shared_ptr(0));
        if (!weights || weights->get_element_type() != ov::element::u3) {
            continue;
        }
        const auto zero_point = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{}, {int8_t{4}});
        subtract->input(1).replace_source_output(
            std::make_shared<ov::op::v0::Convert>(zero_point, subtract->get_input_element_type(1)));
    }
    model->validate_nodes_and_infer_types();
}

}  // namespace test
}  // namespace ov
