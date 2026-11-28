// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "openvino/openvino.hpp"

namespace ov::test {

// Check deterministic, stateless graphs with f32 outputs. Compile the reference independently
// so an in-place write in the reused request cannot also corrupt the reference.
inline void check_reused_request(ov::Core& core,
                                 const std::shared_ptr<ov::Model>& model,
                                 const ov::AnyMap& config,
                                 ov::CompiledModel& compiled,
                                 const std::vector<ov::Tensor>& inputs_a,
                                 const std::vector<ov::Tensor>& inputs_b) {
    auto reused = compiled.create_infer_request();
    size_t call = 0;
    for (const auto* inputs : {&inputs_a, &inputs_b, &inputs_a}) {
        SCOPED_TRACE(::testing::Message() << "Inference call " << ++call);
        auto fresh = core.compile_model(model, "CPU", config).create_infer_request();
        for (size_t i = 0; i < inputs->size(); ++i) {
            fresh.set_input_tensor(i, inputs->at(i));
            reused.set_input_tensor(i, inputs->at(i));
        }
        fresh.infer();
        reused.infer();
        for (size_t i = 0; i < model->outputs().size(); ++i) {
            SCOPED_TRACE(::testing::Message() << "Output " << i);
            const auto expected = fresh.get_output_tensor(i);
            const auto actual = reused.get_output_tensor(i);
            ASSERT_EQ(actual.get_shape(), expected.get_shape());
            ASSERT_EQ(actual.get_element_type(), ov::element::f32);
            ASSERT_EQ(expected.get_element_type(), ov::element::f32);
            const auto* actual_values = actual.data<const float>();
            const auto* expected_values = expected.data<const float>();
            float max_error = 0;
            size_t worst_element = 0;
            for (size_t j = 0; j < actual.get_size(); ++j) {
                const float error = std::abs(actual_values[j] - expected_values[j]);
                ASSERT_TRUE(std::isfinite(error)) << "Non-finite difference at element " << j;
                if (error > max_error) {
                    max_error = error;
                    worst_element = j;
                }
            }
            ASSERT_EQ(max_error, 0.f) << "Largest absolute difference at element " << worst_element;
        }
    }
}

}  // namespace ov::test
