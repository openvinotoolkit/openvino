// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "custom_shape_infer.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/fully_connected.hpp"

namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;

TEST(CpuShapeInfer, FC_InputSize_2) {
    auto activate = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto weight = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, 6});
    auto op = std::make_shared<ov::op::internal::FullyConnected>(
        activate,
        weight,
        std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));
    std::vector<StaticShape> static_input_shapes = {StaticShape{720, 640}, {5, 6}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{720, 5}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(CpuShapeInfer, FC_broadcastWeights1) {
    auto activate = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, -1});
    auto weight = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, 6});
    auto op = std::make_shared<ov::op::internal::FullyConnected>(
        activate,
        weight,
        std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 720, 6}, {5, 6}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 720, 5}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(CpuShapeInfer, FC_broadcastWeights2) {
    auto activate = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto weight = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{5, 6});
    auto op = std::make_shared<ov::op::internal::FullyConnected>(
        activate,
        weight,
        std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));
    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 720, 6}, {5, 6}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{2, 3, 720, 5}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(CpuShapeInfer, FC_broadcastActivations1) {
    auto activate = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{720, -1});
    auto weight = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 5, 6});
    auto op = std::make_shared<ov::op::internal::FullyConnected>(
        activate,
        weight,
        std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));
    std::vector<StaticShape> static_input_shapes = {StaticShape{720, 6}, {1, 5, 6}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 720, 5}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(CpuShapeInfer, FC_broadcastActivations2) {
    auto activate = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto weight = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 1, 5, 6});
    auto op = std::make_shared<ov::op::internal::FullyConnected>(
        activate,
        weight,
        std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));
    std::vector<StaticShape> static_input_shapes = {StaticShape{720, 6}, {1, 1, 5, 6}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 1, 720, 5}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

}  // namespace cpu_shape_infer
}  // namespace unit_test
}  // namespace intel_cpu
}  // namespace ov
