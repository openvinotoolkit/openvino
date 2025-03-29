// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "custom_shape_infer.hpp"
#include "openvino/op/ops.hpp"
namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;

TEST(CpuShapeInfer, UnaryEltwiseTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto node = std::make_shared<ov::op::v0::Relu>(data);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}},
        static_output_shapes = {StaticShape{3, 6, 5, 5}};

    unit_test::cpu_test_shape_infer(node.get(), static_input_shapes, static_output_shapes);
}
} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov

