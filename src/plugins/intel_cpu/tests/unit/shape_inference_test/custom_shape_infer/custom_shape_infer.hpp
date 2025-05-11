// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "cpu_types.h"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/static_shape.hpp"

#pragma once

namespace ov {
namespace intel_cpu {
namespace unit_test {
void cpu_test_shape_infer(ov::Node* op,
                          const std::vector<StaticShape>& input_shapes,
                          std::vector<StaticShape>& output_shapes,
                          const std::unordered_map<size_t, ov::Tensor>& constant_data = {});

using ShapeVector = std::vector<ov::intel_cpu::StaticShape>;

template <class TOp>
class OpCpuShapeInferenceTest : public testing::Test {
protected:
    using op_type = TOp;

    ShapeVector input_shapes, output_shapes;
    ov::intel_cpu::StaticShape exp_shape;
    std::shared_ptr<TOp> op;

    template <class... Args>
    std::shared_ptr<TOp> make_op(Args&&... args) {
        return std::make_shared<TOp>(std::forward<Args>(args)...);
    }
};

std::string boolToString(const bool value);
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
