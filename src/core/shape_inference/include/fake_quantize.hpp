// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <openvino/op/fake_quantize.hpp>

#include "utils.hpp"

template <class T>
void shape_infer(const ov::op::v0::FakeQuantize* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 5 && output_shapes.size() == 1);

    T data_pshape = input_shapes[0];
    ov::op::AutoBroadcastSpec auto_broadcast = op->get_auto_broadcast();

    for (size_t i = 1; i <= 4; ++i) {
        if (auto_broadcast.m_type == ov::op::AutoBroadcastType::NONE) {
            NODE_VALIDATION_CHECK(op, T::merge_into(data_pshape, input_shapes[i]), "Argument shapes are inconsistent.");
        } else if (auto_broadcast.m_type == ov::op::AutoBroadcastType::NUMPY ||
                   auto_broadcast.m_type == ov::op::AutoBroadcastType::PDPD) {
            NODE_VALIDATION_CHECK(op,
                                  T::broadcast_merge_into(data_pshape, input_shapes[i], auto_broadcast),
                                  "Argument shapes are inconsistent.");
        } else {
            NODE_VALIDATION_CHECK(op, false, "Unsupported auto broadcast specification");
        }
    }
    // NOTE: kept as first shape passthrough as by spec we have uni-directional broadcasting
    // meaning that limit inputs do not affect output shape
    // BUT: will not fail in the case of
    // input[0].shape = [1, 3, 1, 1]
    // input[1].shape = [1, 3, 4, 5]
    // This controversial behavior is kept here due to backward-compatibility and the fact that
    // frameworks do not allow such behavior too -- so the chance to have such FQ configuration is minimal
    first_input_passthrough_infer(op, input_shapes, output_shapes);
}