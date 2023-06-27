// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_types.h"
#include <utils/shape_inference/shape_inference_cpu.hpp>
#include <utils/shape_inference/static_shape.hpp>
#include <gmock/gmock.h>

#pragma once

namespace ov {
namespace intel_cpu {
namespace unit_test {
void cpu_test_shape_infer(ov::Node* op,
                     const std::vector<StaticShape>& input_shapes,
                     std::vector<StaticShape>& output_shapes,
                     const std::map<size_t, HostTensorPtr>& constant_data = {});

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
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
