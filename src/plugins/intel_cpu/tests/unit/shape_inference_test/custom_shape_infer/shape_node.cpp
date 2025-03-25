// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ostream>

#include "common_test_utils/test_assertions.hpp"
#include "custom_shape_infer.hpp"
#include "openvino/op/ops.hpp"
namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

TEST(CpuShapeInfer, ShapeOf5DTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto shapeof =
            std::make_shared<op::v0::ShapeOf>(data);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 4, 5, 6}},
            static_output_shapes = {StaticShape{5}};
    unit_test::cpu_test_shape_infer(shapeof.get(), static_input_shapes, static_output_shapes);
}

TEST(CpuShapeInfer, v3ShapeOf5DTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto shapeof =
            std::make_shared<op::v3::ShapeOf>(data);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 4, 5, 6}},
            static_output_shapes = {StaticShape{5}};
    unit_test::cpu_test_shape_infer(shapeof.get(), static_input_shapes, static_output_shapes);
}

} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov

