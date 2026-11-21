// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <transformations/cpu_opset/common/pass/avg_pool_ceil_torch_divisor.hpp>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;
using ov::op::RoundingType;

namespace {

std::shared_ptr<ov::op::v16::AvgPool> make_pool(const ov::Output<ov::Node>& data,
                                                const ov::Shape& kernel,
                                                const ov::Strides& strides,
                                                const ov::Shape& pads,
                                                bool exclude_pad,
                                                RoundingType rounding,
                                                ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT,
                                                const ov::Strides& dilations = {}) {
    return std::make_shared<ov::op::v16::AvgPool>(data,
                                                  strides,
                                                  dilations,
                                                  pads,
                                                  pads,
                                                  kernel,
                                                  exclude_pad,
                                                  rounding,
                                                  auto_pad);
}

}  // namespace

// The case from the issue: 1x1x2x4 input, kernel 2, stride 2, pads_begin {0,1}, pads_end {0,0}.
// W windows start at -1, 1, 3. The last one covers input 3 and position 4, which is beyond pads_end (overhang).
// ONNX / PyTorch divide it by 1 (the overhang is not counted), exclude_pad also by 1 -> ratio 1.
// The first window covers pad -1 and input 0: ONNX / PyTorch divide by 2, exclude_pad by 1 -> ratio 1/2.
TEST_F(TransformationTestsF, AvgPoolCeilTorchDivisor_StaticOverhang) {
    manager.register_pass<ov::intel_cpu::AvgPoolCeilTorchDivisor>();
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 2, 4});
        auto pool = std::make_shared<ov::op::v14::AvgPool>(data,
                                                           ov::Strides{2, 2},
                                                           ov::Shape{0, 1},
                                                           ov::Shape{0, 0},
                                                           ov::Shape{2, 2},
                                                           false,
                                                           RoundingType::CEIL_TORCH);
        model = std::make_shared<ov::Model>(ov::OutputVector{pool}, ov::ParameterVector{data});
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 2, 4});
        auto pool = std::make_shared<ov::op::v16::AvgPool>(data,
                                                           ov::Strides{2, 2},
                                                           ov::Strides{1, 1},
                                                           ov::Shape{0, 1},
                                                           ov::Shape{0, 0},
                                                           ov::Shape{2, 2},
                                                           true,
                                                           RoundingType::CEIL_TORCH);
        auto ratio = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 3}, {0.5F, 1.0F, 1.0F});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(pool, ratio);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{multiply}, ov::ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

// Pools the pass must leave untouched.
struct UntouchedCase {
    std::string name;
    ov::PartialShape shape;
    ov::element::Type type;
    ov::Shape kernel;
    ov::Strides strides;
    ov::Shape pads;
    bool exclude_pad;
    RoundingType rounding;
    ov::op::PadType auto_pad;
};

class AvgPoolCeilTorchDivisorUntouched : public TransformationTestsF, public WithParamInterface<UntouchedCase> {};

TEST_P(AvgPoolCeilTorchDivisorUntouched, CompareWithRef) {
    const auto& param = GetParam();
    manager.register_pass<ov::intel_cpu::AvgPoolCeilTorchDivisor>();
    const auto build = [&param]() {
        auto data = std::make_shared<ov::op::v0::Parameter>(param.type, param.shape);
        auto pool =
            make_pool(data, param.kernel, param.strides, param.pads, param.exclude_pad, param.rounding, param.auto_pad);
        return std::make_shared<ov::Model>(ov::OutputVector{pool}, ov::ParameterVector{data});
    };
    model = build();
    model_ref = build();
}

const auto explicit_pad = ov::op::PadType::EXPLICIT;
const std::vector<UntouchedCase> untouched_cases = {
    // already exclude_pad: every executor is correct
    {"ExcludePad",
     {1, 3, 6, 6},
     ov::element::f32,
     {3, 3},
     {3, 3},
     {1, 1},
     true,
     RoundingType::CEIL_TORCH,
     explicit_pad},
    // plain CEIL keeps its own semantics
    {"CeilRounding", {1, 3, 6, 6}, ov::element::f32, {3, 3}, {3, 3}, {1, 1}, false, RoundingType::CEIL, explicit_pad},
    {"FloorRounding", {1, 3, 6, 6}, ov::element::f32, {3, 3}, {3, 3}, {1, 1}, false, RoundingType::FLOOR, explicit_pad},
    // without padding include_pad and exclude_pad are the same
    {"NoPads", {1, 3, 7, 7}, ov::element::f32, {2, 2}, {2, 2}, {0, 0}, false, RoundingType::CEIL_TORCH, explicit_pad},
    // 7x7, k3 s2 p1: last window ends exactly at size + pad_end, no overhang
    {"NoOverhang",
     {1, 3, 7, 7},
     ov::element::f32,
     {3, 3},
     {2, 2},
     {1, 1},
     false,
     RoundingType::CEIL_TORCH,
     explicit_pad},
    // auto padding resolves pads itself, there is no ceil overhang
    {"SameUpper",
     {1, 3, 6, 6},
     ov::element::f32,
     {3, 3},
     {3, 3},
     {0, 0},
     false,
     RoundingType::CEIL_TORCH,
     ov::op::PadType::SAME_UPPER},
    // padding >= kernel: an exclude_pad pool would be invalid, so the pool is kept
    {"PadNotSmallerThanKernelDynamic",
     {1, 3, -1, -1},
     ov::element::f32,
     {2, 2},
     {2, 2},
     {2, 2},
     false,
     RoundingType::CEIL_TORCH,
     explicit_pad},
    // integer pools are not rewritten (the ratio would be truncated)
    {"Integer", {1, 3, 6, 6}, ov::element::i32, {3, 3}, {3, 3}, {1, 1}, false, RoundingType::CEIL_TORCH, explicit_pad},
    // unknown rank: spatial axes unknown
    {"DynamicRank",
     ov::PartialShape::dynamic(),
     ov::element::f32,
     {3, 3},
     {3, 3},
     {1, 1},
     false,
     RoundingType::CEIL_TORCH,
     explicit_pad},
};

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         AvgPoolCeilTorchDivisorUntouched,
                         ValuesIn(untouched_cases),
                         [](const TestParamInfo<UntouchedCase>& info) {
                             return info.param.name;
                         });

// Static pool whose last window overhangs, but pads_begin is not smaller than the kernel: the exclude_pad
// rewrite would fail validation, so the pool is kept. 1x1x3 input, k2 s2, pads {2}/{0}: windows start at -2, 0, 2.
TEST_F(TransformationTestsF, AvgPoolCeilTorchDivisor_StaticPadNotSmallerThanKernel) {
    manager.register_pass<ov::intel_cpu::AvgPoolCeilTorchDivisor>();
    const auto build = []() {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 3});
        auto pool = std::make_shared<ov::op::v14::AvgPool>(data,
                                                           ov::Strides{2},
                                                           ov::Shape{2},
                                                           ov::Shape{0},
                                                           ov::Shape{2},
                                                           false,
                                                           RoundingType::CEIL_TORCH);
        return std::make_shared<ov::Model>(ov::OutputVector{pool}, ov::ParameterVector{data});
    };
    model = build();
    model_ref = build();
}

// Dynamic spatial dims: the pool is rewritten to exclude_pad * ratio computed in the graph, with the
// original friendly name kept on the output.
TEST(TransformationTests, AvgPoolCeilTorchDivisor_DynamicSpatial) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, -1, -1});
    auto pool = make_pool(data, {2, 2}, {2, 2}, {1, 1}, false, RoundingType::CEIL_TORCH);
    pool->set_friendly_name("pool");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{pool}, ov::ParameterVector{data});

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::AvgPoolCeilTorchDivisor>();
    manager.run_passes(model);

    const auto result = model->get_results()[0]->get_input_node_shared_ptr(0);
    ASSERT_TRUE(ov::is_type<ov::op::v1::Multiply>(result));
    EXPECT_EQ(result->get_friendly_name(), "pool");
    const auto exclude_pool = ov::as_type_ptr<ov::op::v16::AvgPool>(result->get_input_node_shared_ptr(0));
    ASSERT_NE(exclude_pool, nullptr);
    EXPECT_TRUE(exclude_pool->get_exclude_pad());
    EXPECT_EQ(exclude_pool->get_rounding_type(), RoundingType::CEIL_TORCH);
    EXPECT_EQ(result->get_output_partial_shape(0), pool->get_output_partial_shape(0));
    // the pass must not match its own output again
    manager.run_passes(model);
    EXPECT_EQ(model->get_results()[0]->get_input_node_shared_ptr(0), result);
}
