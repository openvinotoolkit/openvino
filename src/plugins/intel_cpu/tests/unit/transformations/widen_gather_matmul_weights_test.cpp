// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/widen_gather_matmul_weights.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/gather_matmul.hpp"

using namespace ov::intel_cpu;

namespace {

// GatherMatmul's x86_64 capability list (see GatherMatmul::getSupportedCompressedWeightsTypes):
// no u2, hence the pass.
const std::vector<ov::element::Type> kSupported{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4};

constexpr size_t kExperts = 4;
constexpr size_t kOc = 8;
constexpr size_t kGroups = 2;
constexpr size_t kGroupSize = 16;
constexpr size_t kIc = kGroups * kGroupSize;

std::vector<int32_t> weight_values() {
    std::vector<int32_t> v(kExperts * kOc * kIc);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<int32_t>(i % 4);  // covers the full u2 range
    }
    return v;
}

// weights(`wt`, grouped) -> Convert -> Subtract(zp) -> Multiply(scale) -> Reshape: a
// CompressedWeightsBlock over per-expert weights.
std::shared_ptr<ov::Node> make_weights_block(ov::element::Type wt, const std::vector<int32_t>& values) {
    const ov::Shape grouped{kExperts, kOc, kGroups, kGroupSize};
    const ov::Shape per_group{kExperts, kOc, kGroups, 1};
    auto weights = std::make_shared<ov::op::v0::Constant>(wt, grouped, values);
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    auto zp = ov::op::v0::Constant::create(ov::element::u8, per_group, {1});
    auto sub =
        std::make_shared<ov::op::v1::Subtract>(convert, std::make_shared<ov::op::v0::Convert>(zp, ov::element::f32));
    auto scale = ov::op::v0::Constant::create(ov::element::f32, per_group, {0.5f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(sub, scale);
    auto pattern =
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{kExperts, kOc, kIc});
    return std::make_shared<ov::op::v1::Reshape>(mul, pattern, false);
}

std::shared_ptr<ov::Model> make_gather_matmul_model(ov::element::Type wt, const std::vector<int32_t>& values) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 6, kIc});
    auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{6, 2}, {1});
    auto bgm = std::make_shared<ov::op::internal::GatherMatmul>(input, make_weights_block(wt, values), index);
    return std::make_shared<ov::Model>(ov::OutputVector{bgm}, ov::ParameterVector{input});
}

void run_pass(const std::shared_ptr<ov::Model>& model, const std::vector<ov::element::Type>& supported) {
    ov::pass::Manager manager;
    manager.register_pass<WidenGatherMatmulWeights>(supported);
    manager.run_passes(model);
}

// The weight Constant at the head of the decompression subgraph: the one feeding the Convert.
std::shared_ptr<ov::op::v0::Constant> find_weights(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::op::v0::Constant> found;
    for (const auto& node : model->get_ordered_ops()) {
        auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node);
        if (!convert) {
            continue;
        }
        auto c = ov::as_type_ptr<ov::op::v0::Constant>(convert->get_input_node_shared_ptr(0));
        if (c && c->get_shape() == ov::Shape{kExperts, kOc, kGroups, kGroupSize}) {
            EXPECT_EQ(found, nullptr) << "expected exactly one weight Constant";
            found = c;
        }
    }
    return found;
}

}  // namespace

// A u2 expert-weight Constant is re-emitted as u4, values preserved. Only the storage type
// changes: the rest of the decompression subgraph is untouched, so the compression pass that runs
// next can match it.
TEST(WidenGatherMatmulWeights, U2WeightsWidenedToU4) {
    const auto values = weight_values();
    auto model = make_gather_matmul_model(ov::element::u2, values);
    const size_t ops_before = model->get_ordered_ops().size();

    run_pass(model, kSupported);

    auto weights = find_weights(model);
    ASSERT_NE(weights, nullptr);
    EXPECT_EQ(weights->get_element_type(), ov::element::u4);
    EXPECT_EQ(weights->get_shape(), (ov::Shape{kExperts, kOc, kGroups, kGroupSize}));
    EXPECT_EQ(weights->cast_vector<int32_t>(), values) << "widening must be lossless";
    EXPECT_EQ(model->get_ordered_ops().size(), ops_before) << "no node added or removed";
}

// u4 is natively supported: nothing to do.
TEST(WidenGatherMatmulWeights, SupportedTypeIsUntouched) {
    auto model = make_gather_matmul_model(ov::element::u4, weight_values());
    run_pass(model, kSupported);
    EXPECT_EQ(find_weights(model)->get_element_type(), ov::element::u4);
}

// The pass is driven by the node's own capability list, so on a build where u2 is supported it is
// a no-op rather than a wasted 2x.
TEST(WidenGatherMatmulWeights, NoOpWhenNarrowTypeIsSupported) {
    auto model = make_gather_matmul_model(ov::element::u2, weight_values());
    std::vector<ov::element::Type> supported_with_u2{kSupported};
    supported_with_u2.push_back(ov::element::u2);

    run_pass(model, supported_with_u2);

    EXPECT_EQ(find_weights(model)->get_element_type(), ov::element::u2);
}

// Widening costs 2x the weight bytes, so it must not touch weights that no GatherMatmul consumes:
// those reach FullyConnected, which supports u2 natively.
TEST(WidenGatherMatmulWeights, NonGatherMatmulConsumerIsUntouched) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{kExperts, 6, kIc});
    auto mm = std::make_shared<ov::op::v0::MatMul>(input,
                                                  make_weights_block(ov::element::u2, weight_values()),
                                                  false,
                                                  true);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mm}, ov::ParameterVector{input});

    run_pass(model, kSupported);

    EXPECT_EQ(find_weights(model)->get_element_type(), ov::element::u2);
}
