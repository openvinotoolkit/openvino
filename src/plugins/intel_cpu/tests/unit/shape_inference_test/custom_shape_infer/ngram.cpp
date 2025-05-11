// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "transformations/cpu_opset/common/op/ngram.hpp"
#include "custom_shape_infer.hpp"

namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;

TEST(CpuShapeInfer, Ngram) {
    auto embeddings = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto idces = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto op = std::make_shared<ov::intel_cpu::NgramNode>(embeddings, idces, 3);
    std::vector<StaticShape> static_input_shapes = {StaticShape{720, 640}, {5, 6}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{720, 640 * 3}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}
} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov

