// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "partitioning/patterns/opt.hpp"
#include "util.hpp"

namespace {

using namespace ov;
using namespace ov::npuw::patterns::opt;

constexpr size_t kVocab = 16;
constexpr size_t kGroups = 4;
constexpr size_t kGroupSize = 512;  // the host gather requires the embedding size >= 2048
constexpr size_t kTokens = 3;

std::shared_ptr<op::v0::Constant> axis0() {
    return op::v0::Constant::create(element::i32, Shape{}, {0});
}

size_t count_ops(const std::shared_ptr<ov::Model>& model, const std::string& type_name) {
    const auto ops = model->get_ordered_ops();
    return std::count_if(ops.begin(), ops.end(), [&](const std::shared_ptr<Node>& node) {
        return node->get_type_name() == type_name;
    });
}

// Const(W u2)[V,G,GS] -> Convert(f16) -> [Subtract(Convert(f16)(Const(Z u2)[V,G,1]))] -> Multiply(Const(S f16)[V,G,1])
//   -> Reshape[V,H] -> Convert(f32) -> Gather(Param(ids)[1,N], axis 0)
std::shared_ptr<ov::Model> make_group_quantized_vocab_model(const bool with_zero_point,
                                                            const Shape& zero_point_shape = Shape{kVocab, kGroups, 1}) {
    auto weight = op::v0::Constant::create(element::u2,
                                           Shape{kVocab, kGroups, kGroupSize},
                                           std::vector<uint8_t>(kVocab * kGroups * kGroupSize, 1));
    auto scale = op::v0::Constant::create(element::f16,
                                          Shape{kVocab, kGroups, 1},
                                          std::vector<ov::float16>(kVocab * kGroups, ov::float16(0.5f)));
    std::shared_ptr<Node> dq = std::make_shared<op::v0::Convert>(weight, element::f16);
    if (with_zero_point) {
        auto zerop = op::v0::Constant::create(element::u2,
                                              zero_point_shape,
                                              std::vector<uint8_t>(shape_size(zero_point_shape), 2));
        dq = std::make_shared<op::v1::Subtract>(dq, std::make_shared<op::v0::Convert>(zerop, element::f16));
    }
    dq = std::make_shared<op::v1::Multiply>(dq, scale);
    auto reshape = std::make_shared<op::v1::Reshape>(
        dq,
        op::v0::Constant::create(element::i32, Shape{2}, std::vector<size_t>{kVocab, kGroups * kGroupSize}),
        false);
    auto cvt = std::make_shared<op::v0::Convert>(reshape, element::f32);
    auto ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, kTokens});
    auto gather = std::make_shared<op::v8::Gather>(cvt, ids, axis0());
    return std::make_shared<ov::Model>(OutputVector{gather}, ParameterVector{ids});
}

// Param(W u2)[V,G,GS] -> Convert(f16) -> Gather -.
// Param(Z)[V,G,1]     -> Convert(f16) -> Gather -> Subtract -.
// Param(S f16)[V,G,1] -----------------> Gather ------------> Multiply -> Reshape[1,N,H] -> Convert(f32)
std::shared_ptr<ov::Model> make_host_gather_group_zero_point_model(const element::Type zero_point_type,
                                                                   const element::Type scale_type = element::f16) {
    auto ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, kTokens});
    auto w = std::make_shared<op::v0::Parameter>(element::u2, Shape{kVocab, kGroups, kGroupSize});
    auto z = std::make_shared<op::v0::Parameter>(zero_point_type, Shape{kVocab, kGroups, 1});
    auto s = std::make_shared<op::v0::Parameter>(scale_type, Shape{kVocab, kGroups, 1});

    auto gw = std::make_shared<op::v8::Gather>(std::make_shared<op::v0::Convert>(w, element::f16), ids, axis0());
    auto gz = std::make_shared<op::v8::Gather>(std::make_shared<op::v0::Convert>(z, element::f16), ids, axis0());
    std::shared_ptr<Node> scale = s;
    if (scale_type != element::f16) {
        scale = std::make_shared<op::v0::Convert>(s, element::f16);
    }
    auto gs = std::make_shared<op::v8::Gather>(scale, ids, axis0());
    auto mul = std::make_shared<op::v1::Multiply>(std::make_shared<op::v1::Subtract>(gw, gz), gs);
    auto reshape = std::make_shared<op::v1::Reshape>(
        mul,
        op::v0::Constant::create(element::i32, Shape{3}, std::vector<size_t>{1, kTokens, kGroups * kGroupSize}),
        false);
    auto out = std::make_shared<op::v0::Convert>(reshape, element::f32);
    return std::make_shared<ov::Model>(ResultVector{std::make_shared<op::v0::Result>(out)},
                                       ParameterVector{ids, w, z, s});
}

template <typename Pass, typename... Args>
void run_pass(const std::shared_ptr<ov::Model>& model, Args&&... args) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<Pass>(std::forward<Args>(args)...);
    rewr.run_on_model(model);
}

}  // namespace

TEST(DQLiftGatherZeroPointTest, LiftsGathersAboveZeroPointDequantization) {
    auto model = make_group_quantized_vocab_model(true);

    run_pass<DQLiftGatherSymGQ>(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops(model, "Gather"), 3u);
    EXPECT_EQ(count_ops(model, "Subtract"), 1u);
    EXPECT_EQ(model->get_results().front()->get_input_shape(0), (Shape{1, kTokens, kGroups * kGroupSize}));
}

TEST(DQLiftGatherZeroPointTest, LiftsGathersAboveSymmetricDequantization) {
    auto model = make_group_quantized_vocab_model(false);

    run_pass<DQLiftGatherSymGQ>(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops(model, "Gather"), 2u);
    EXPECT_EQ(count_ops(model, "Subtract"), 0u);
}

TEST(DQLiftGatherZeroPointTest, DoesNotGatherZeroPointBroadcastOverVocabRows) {
    for (const auto& zero_point_shape : {Shape{}, Shape{1, kGroups, 1}}) {
        auto model = make_group_quantized_vocab_model(true, zero_point_shape);

        run_pass<DQLiftGatherSymGQ>(model);
        model->validate_nodes_and_infer_types();

        EXPECT_EQ(count_ops(model, "Gather"), 2u);
        EXPECT_EQ(count_ops(model, "Subtract"), 1u);
        EXPECT_EQ(model->get_results().front()->get_input_shape(0), (Shape{1, kTokens, kGroups * kGroupSize}));
    }
}

TEST(HostGatherDQZeroPointTest, UnpacksU2VocabWithZeroPointOnHost) {
    auto model = make_host_gather_group_zero_point_model(element::u2);

    Context ctx(model, 0);
    run_pass<HostGatherDQ>(model, std::ref(ctx));

    ASSERT_TRUE(ctx.params_to_gather.has_value());
    ASSERT_EQ(ctx.params_to_unpack.size(), 1u);
    const auto& unpack = ctx.params_to_unpack.begin()->second;
    EXPECT_NE(unpack.w, nullptr);
    EXPECT_NE(unpack.z, nullptr);
    EXPECT_NE(unpack.s, nullptr);
    EXPECT_EQ(ctx.params_to_unpack.begin()->first->get_shape(), (Shape{kVocab, kGroups * kGroupSize}));
    EXPECT_EQ(count_ops(model, "Gather"), 0u);
}

TEST(HostGatherDQZeroPointTest, KeepsGraphWhenZeroPointTypeDiffersFromWeights) {
    auto model = make_host_gather_group_zero_point_model(element::u8);

    Context ctx(model, 0);
    run_pass<HostGatherDQ>(model, std::ref(ctx));

    EXPECT_FALSE(ctx.params_to_gather.has_value());
    EXPECT_EQ(count_ops(model, "Gather"), 3u);
}

TEST(HostGatherDQZeroPointTest, KeepsGraphWhenScaleIsNotF16) {
    auto model = make_host_gather_group_zero_point_model(element::u2, element::f32);

    Context ctx(model, 0);
    run_pass<HostGatherDQ>(model, std::ref(ctx));

    EXPECT_FALSE(ctx.params_to_gather.has_value());
    EXPECT_EQ(count_ops(model, "Gather"), 3u);
}

TEST(UnpackU2Test, DequantizesGroupsWithZeroPointAndScale) {
    constexpr size_t rows = 2;
    constexpr size_t groups = 2;
    constexpr size_t group_size = 8;
    const std::vector<uint8_t> weight_values = {0, 1, 2, 3, 3, 2, 1, 0, 1, 1, 2, 2, 3, 3, 0, 0,
                                                2, 3, 0, 1, 1, 0, 3, 2, 0, 3, 1, 2, 2, 1, 3, 0};
    const std::vector<uint8_t> zero_point_values = {1, 2, 0, 3};
    const std::vector<float> scale_values = {0.5f, 0.25f, 2.0f, 1.0f};

    const auto pack_u2 = [](const std::vector<uint8_t>& values, ov::Tensor& tensor) {
        auto* bytes = static_cast<uint8_t*>(tensor.data());
        std::fill_n(bytes, tensor.get_byte_size(), uint8_t{0});
        for (size_t i = 0; i < values.size(); ++i) {
            bytes[i / 4] |= static_cast<uint8_t>(values[i] << (2 * (i % 4)));
        }
    };
    ov::Tensor weights(element::u2, Shape{rows, groups, group_size});
    ov::Tensor zero_points(element::u2, Shape{rows, groups, 1});
    ov::Tensor scales(element::f16, Shape{rows, groups, 1});
    ov::Tensor result(element::f16, Shape{rows, groups * group_size});
    pack_u2(weight_values, weights);
    pack_u2(zero_point_values, zero_points);
    std::transform(scale_values.begin(), scale_values.end(), scales.data<ov::float16>(), [](const float value) {
        return ov::float16(value);
    });

    ov::npuw::util::unpack(ov::get_tensor_impl(weights),
                           ov::get_tensor_impl(zero_points),
                           ov::get_tensor_impl(scales),
                           ov::get_tensor_impl(result));

    const auto* actual = result.data<const ov::float16>();
    for (size_t i = 0; i < weight_values.size(); ++i) {
        const size_t group = i / group_size;
        const float expected = (static_cast<float>(weight_values[i]) - zero_point_values[group]) * scale_values[group];
        EXPECT_EQ(static_cast<float>(actual[i]), expected) << "element " << i;
    }
}
