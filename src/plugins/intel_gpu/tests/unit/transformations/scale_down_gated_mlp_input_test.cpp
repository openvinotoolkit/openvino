// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/scale_down_gated_mlp_input.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/glu.hpp"
#include "transformations/common_optimizations/glu_fusion.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

namespace {
// FullyConnectedCompressed with N output channels whose decompression scales are `scales` (one per channel).
std::shared_ptr<ov::Node> fc(const ov::Output<ov::Node>& in, size_t n, size_t k, const std::vector<float>& scales, float bias = 0.f) {
    auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{n, k}, {1});
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{n, 1}, scales);
    auto zp = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{n, 1}, {1});
    std::shared_ptr<ov::Node> b = std::make_shared<ov::intel_gpu::op::Placeholder>();
    if (bias != 0.f)
        b = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, n}, {bias});
    return std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(in, weights, b, scale, zp);
}

std::vector<float> filled(size_t n, float value) {
    return std::vector<float>(n, value);
}

std::shared_ptr<ov::op::v0::Parameter> input() {
    return std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
}

std::shared_ptr<ov::Model> model_of(const std::shared_ptr<ov::Node>& out, const std::shared_ptr<ov::op::v0::Parameter>& in) {
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(out)}, ov::ParameterVector{in});
}

// in -> FC(gate) -> Swish -> Multiply(., FC(up)) -> FC(down); `fixed` builds the expected scales.
std::shared_ptr<ov::Model> gated_mlp(bool fixed, bool swap, bool down_bias = false) {
    auto in = input();
    auto gate = std::make_shared<ov::op::v4::Swish>(fc(in, 64, 16, filled(64, 1.f)));
    auto up = fc(in, 64, 16, filled(64, fixed ? 0.125f : 1.f));
    auto product = swap ? std::make_shared<ov::op::v1::Multiply>(up, gate) : std::make_shared<ov::op::v1::Multiply>(gate, up);
    return model_of(fc(product, 16, 64, filled(16, fixed ? 16.f : 2.f), down_bias ? 0.5f : 0.f), in);
}

// in -> FC(gate|up merged) -> VariadicSplit -> Multiply(Swish(gate), up) -> FC(down), as after FullyConnectedHorizontalFusion.
std::shared_ptr<ov::Model> merged_gated_mlp(bool fixed, bool up_first) {
    auto in = input();
    std::vector<float> scales = filled(128, 1.f);
    if (fixed)
        std::fill(scales.begin() + (up_first ? 0 : 64), scales.begin() + (up_first ? 64 : 128), 0.125f);
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(fc(in, 128, 16, scales),
                                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {64, 64}));
    auto gate = std::make_shared<ov::op::v4::Swish>(split->output(up_first ? 1 : 0));
    auto product = std::make_shared<ov::op::v1::Multiply>(gate, split->output(up_first ? 0 : 1));
    return model_of(fc(product, 16, 64, filled(16, fixed ? 16.f : 2.f)), in);
}
}  // namespace

TEST_F(TransformationTestsF, ScaleDownGatedMLPInput) {
    model = gated_mlp(false, false);
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = gated_mlp(true, false);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_SwappedOperands) {
    model = gated_mlp(false, true);
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = gated_mlp(true, true);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

// The down projection output keeps its magnitude, so its bias is left as is.
TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_BiasOnDown) {
    model = gated_mlp(false, false, true);
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = gated_mlp(true, false, true);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_MergedGateUp) {
    model = merged_gated_mlp(false, false);
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = merged_gated_mlp(true, false);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_MergedUpFirst) {
    model = merged_gated_mlp(false, true);
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = merged_gated_mlp(true, true);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

// The pass adds no node, so the merged gate/up pair still fuses into GLU.
TEST(ScaleDownGatedMLPInputTest, MergedGateUpKeepsGLUFusion) {
    auto model = merged_gated_mlp(false, false);
    ov::pass::Manager manager;
    manager.register_pass<ScaleDownGatedMLPInput>();
    manager.register_pass<ov::pass::GLUFusion>();
    manager.run_passes(model);
    size_t glu_count = 0;
    for (const auto& op : model->get_ops())
        glu_count += ov::is_type<ov::op::internal::GLU>(op) ? 1 : 0;
    ASSERT_EQ(glu_count, 1u);
}

// A bias on the up projection is scaled together with its decompression scales.
TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_BiasOnUp) {
    auto build = [](bool fixed) {
        auto in = input();
        auto gate = std::make_shared<ov::op::v4::Swish>(fc(in, 64, 16, filled(64, 1.f)));
        auto up = fc(in, 64, 16, filled(64, fixed ? 0.125f : 1.f), fixed ? 0.0625f : 0.5f);
        return model_of(fc(std::make_shared<ov::op::v1::Multiply>(gate, up), 16, 64, filled(16, fixed ? 16.f : 2.f)), in);
    };
    model = build(false);
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = build(true);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

// The factor passes through a Reshape and scales the constant of an unfused bias Add on the up branch.
TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_UpThroughReshapeAndAdd) {
    auto build = [](bool fixed) {
        auto in = input();
        auto gate = std::make_shared<ov::op::v4::Swish>(fc(in, 64, 16, filled(64, 1.f)));
        auto reshape = std::make_shared<ov::op::v1::Reshape>(fc(in, 64, 16, filled(64, fixed ? 0.125f : 1.f)),
                                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 64}),
                                                             true);
        auto up = std::make_shared<ov::op::v1::Add>(reshape, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{64}, {fixed ? 0.25f : 2.f}));
        return model_of(fc(std::make_shared<ov::op::v1::Multiply>(gate, up), 16, 64, filled(16, fixed ? 16.f : 2.f)), in);
    };
    model = build(false);
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = build(true);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

// 1e-4 / 8 is below the smallest normal FP16 value, so the scaling would not be exact.
TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_SubnormalScale_NotApplied) {
    auto build = [] {
        auto in = input();
        auto gate = std::make_shared<ov::op::v4::Swish>(fc(in, 64, 16, filled(64, 1.f)));
        auto product = std::make_shared<ov::op::v1::Multiply>(gate, fc(in, 64, 16, filled(64, 1e-4f)));
        return model_of(fc(product, 16, 64, filled(16, 2.f)), in);
    };
    model = build();
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = build();
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_NoSwish_NotApplied) {
    auto build = [] {
        auto in = input();
        auto product = std::make_shared<ov::op::v1::Multiply>(fc(in, 64, 16, filled(64, 1.f)), fc(in, 64, 16, filled(64, 1.f)));
        return model_of(fc(product, 16, 64, filled(16, 2.f)), in);
    };
    model = build();
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = build();
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_ProductWithTwoConsumers_NotApplied) {
    auto build = [] {
        auto in = input();
        auto gate = std::make_shared<ov::op::v4::Swish>(fc(in, 64, 16, filled(64, 1.f)));
        auto product = std::make_shared<ov::op::v1::Multiply>(gate, fc(in, 64, 16, filled(64, 1.f)));
        return std::make_shared<ov::Model>(
            ov::ResultVector{std::make_shared<ov::op::v0::Result>(fc(product, 16, 64, filled(16, 2.f))), std::make_shared<ov::op::v0::Result>(product)},
            ov::ParameterVector{in});
    };
    model = build();
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = build();
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
