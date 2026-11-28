// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_grouped_matmul_to_matmul.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
namespace v17 = ov::op::v17;

namespace {

constexpr auto f32 = ov::element::f32;
constexpr auto i32 = ov::element::i32;
constexpr auto i64 = ov::element::i64;

// Reference decomposition of the 2Dx3D (offsets) case into per-group MatMuls.
ov::Output<ov::Node> ref_2dx3d_decomposition(const ov::Output<ov::Node>& mat_a,
                                             const ov::Output<ov::Node>& mat_b,
                                             const ov::Output<ov::Node>& offsets,
                                             int64_t num_groups,
                                             const ov::OutputVector& compressed_weights = {}) {
    const auto offsets_et = offsets.get_element_type();

    auto axis_scalar = v0::Constant::create(i32, ov::Shape{}, {0});
    auto axis_1d = v0::Constant::create(i32, ov::Shape{1}, {0});

    ov::Output<ov::Node> split_lengths = offsets;
    if (num_groups > 1) {
        auto zero = v0::Constant::create(offsets_et, ov::Shape{1}, {0});
        auto head_stop = v0::Constant::create(i32, ov::Shape{1}, {num_groups - 1});
        auto step = v0::Constant::create(i32, ov::Shape{1}, {1});
        auto offsets_head = std::make_shared<v8::Slice>(offsets, zero, head_stop, step, axis_1d);
        auto shifted = std::make_shared<v0::Concat>(ov::OutputVector{zero, offsets_head}, 0);
        split_lengths = std::make_shared<v1::Subtract>(offsets, shifted);
    }

    auto a_split = std::make_shared<v1::VariadicSplit>(mat_a, axis_scalar, split_lengths);
    auto b_split = std::make_shared<v1::Split>(mat_b, axis_scalar, num_groups);

    ov::OutputVector group_outputs;
    for (int64_t g = 0; g < num_groups; ++g) {
        ov::Output<ov::Node> b_g = compressed_weights.empty()
                                       ? std::make_shared<v0::Squeeze>(b_split->output(g), axis_1d)->output(0)
                                       : compressed_weights[g];
        auto mm = std::make_shared<v0::MatMul>(a_split->output(g), b_g, /*transpose_a=*/false, /*transpose_b=*/true);
        group_outputs.push_back(mm);
    }
    return std::make_shared<v0::Concat>(group_outputs, 0);
}

}  // namespace

// 2Dx3D with dynamic group dimension: cannot decompose into a fixed number of MatMuls
TEST_F(TransformationTestsF, ConvertGroupedMatMulToMatMul_2Dx3D_DynamicGroupsNoChange) {
    using ov::Dimension;
    using ov::PartialShape;
    constexpr size_t K = 8, N = 16;

    auto mat_a = std::make_shared<v0::Parameter>(f32, PartialShape{Dimension::dynamic(), K});
    auto mat_b = std::make_shared<v0::Parameter>(f32, PartialShape{Dimension::dynamic(), N, K});
    auto offsets = std::make_shared<v0::Parameter>(i32, PartialShape{Dimension::dynamic()});
    auto gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b, offsets);
    model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{mat_a, mat_b, offsets});

    manager.register_pass<ov::pass::ConvertGroupedMatMulToMatMul>();
    // No model_ref: transformation must be a no-op.
}

// {G, M, K, N}
using TestParams3Dx3D = std::tuple<size_t, size_t, size_t, size_t>;
class ConvertGroupedMatMulToMatMul3Dx3DAccuracy : public TransformationTestsF,
                                                  public testing::WithParamInterface<TestParams3Dx3D> {};

TEST_P(ConvertGroupedMatMulToMatMul3Dx3DAccuracy, MatchesReference) {
    const auto [G, M, K, N] = GetParam();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);

    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{G, M, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{G, N, K});
        auto gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b);
        model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{mat_a, mat_b});

        manager.register_pass<ov::pass::ConvertGroupedMatMulToMatMul>();
    }
    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{G, M, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{G, N, K});
        auto mm = std::make_shared<v0::MatMul>(mat_a, mat_b, /*transpose_a=*/false, /*transpose_b=*/true);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{mm}, ov::ParameterVector{mat_a, mat_b});
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes3Dx3D,
                         ConvertGroupedMatMulToMatMul3Dx3DAccuracy,
                         testing::Values(TestParams3Dx3D{1, 8, 16, 8},
                                         TestParams3Dx3D{2, 3, 8, 16},
                                         TestParams3Dx3D{4, 1, 8, 8},
                                         TestParams3Dx3D{3, 7, 16, 8}));

// {T, K, N, cumulative offsets}
using ShapeParams2Dx3D = std::tuple<size_t, size_t, size_t, std::vector<int32_t>>;
// shapes + offsets element type
using TestParams2Dx3D = std::tuple<ShapeParams2Dx3D, ov::element::Type>;

class ConvertGroupedMatMulToMatMul2Dx3DAccuracy : public TransformationTestsF,
                                                  public testing::WithParamInterface<TestParams2Dx3D> {};

TEST_P(ConvertGroupedMatMulToMatMul2Dx3DAccuracy, MatchesReference) {
    const auto& [shapes, offsets_et] = GetParam();
    const auto& [T, K, N, offsets_vec] = shapes;
    const auto G = static_cast<int64_t>(offsets_vec.size());
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);

    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{offsets_vec.size(), N, K});
        auto offsets = v0::Constant::create(offsets_et, ov::Shape{offsets_vec.size()}, offsets_vec);
        auto gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b, offsets);
        model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{mat_a, mat_b});

        manager.register_pass<ov::pass::ConvertGroupedMatMulToMatMul>();
    }
    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{offsets_vec.size(), N, K});
        auto offsets = v0::Constant::create(offsets_et, ov::Shape{offsets_vec.size()}, offsets_vec);
        auto out = ref_2dx3d_decomposition(mat_a, mat_b, offsets, G);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{mat_a, mat_b});
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes2Dx3D,
                         ConvertGroupedMatMulToMatMul2Dx3DAccuracy,
                         testing::Combine(testing::Values(ShapeParams2Dx3D{8, 8, 16, {8}},
                                                          ShapeParams2Dx3D{16, 16, 8, {8, 16}},
                                                          ShapeParams2Dx3D{24, 8, 8, {8, 16, 24}},
                                                          // experts without tokens -> zero-length groups
                                                          ShapeParams2Dx3D{16, 8, 8, {0, 8, 8, 16}}),
                                          testing::Values(i32, i64)));

class ConvertGroupedMatMulToMatMulCompressed
    : public TransformationTestsF,
      public testing::WithParamInterface<std::tuple<ov::element::Type, std::vector<int32_t>, bool>> {};

TEST_P(ConvertGroupedMatMulToMatMulCompressed, PreservesExpertQuantization) {
    const auto weight_type = std::get<0>(GetParam());
    const auto& ends = std::get<1>(GetParam());
    const bool only_compressed_split = std::get<2>(GetParam());
    const size_t groups = ends.size();
    constexpr size_t width = 8, channels = 256;
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    auto dequantize = [&](const ov::Shape& shape, const std::vector<int32_t>& values) {
        auto zero_shape = shape;
        zero_shape.back() = 1;
        auto packed = v0::Constant::create(weight_type, shape, values);
        auto zero = v0::Constant::create(weight_type, zero_shape, {1});
        auto scales = v0::Constant::create(ov::element::f16, zero_shape, {0.125f});
        auto converted = std::make_shared<v0::Convert>(packed, ov::element::f16);
        auto zero_converted = std::make_shared<v0::Convert>(zero, ov::element::f16);
        auto shifted = std::make_shared<v1::Subtract>(converted, zero_converted);
        auto scaled = std::make_shared<v1::Multiply>(shifted, scales);
        std::vector<int64_t> target(shape.begin(), shape.end() - 2);
        target.push_back(channels);
        auto pattern = v0::Constant::create(i64, ov::Shape{target.size()}, target);
        auto reshaped = std::make_shared<v1::Reshape>(scaled, pattern, false);
        return std::make_shared<v0::Convert>(reshaped, f32);
    };
    std::vector<int32_t> packed_values;
    for (size_t expert_index = 0; expert_index < groups; ++expert_index) {
        packed_values.insert(packed_values.end(), width * channels, static_cast<int32_t>(expert_index + 1));
    }
    auto activations = std::make_shared<v0::Parameter>(f32, ov::Shape{static_cast<size_t>(ends.back()), channels});
    auto offsets = v0::Constant::create(i32, ov::Shape{groups}, ends);
    auto weights = dequantize({groups, width, 2, 128}, packed_values);
    auto grouped = std::make_shared<v17::GroupedMatMul>(activations, weights, offsets);
    model = std::make_shared<ov::Model>(ov::OutputVector{grouped}, ov::ParameterVector{activations});
    manager.register_pass<ov::pass::ConvertGroupedMatMulToMatMul>(only_compressed_split);

    ov::OutputVector experts;
    for (size_t expert_index = 0; expert_index < groups; ++expert_index) {
        experts.push_back(dequantize({width, 2, 128}, {static_cast<int32_t>(expert_index + 1)}));
    }
    auto expected = ref_2dx3d_decomposition(activations, weights, offsets, groups, experts);
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{expected}, ov::ParameterVector{activations});
}

TEST_F(TransformationTestsF, ConvertGroupedMatMulToMatMul_2Dx3D_MarkedDequantizationNotSplit) {
    constexpr size_t groups = 2, width = 8, channels = 256;
    const std::vector<int32_t> ends{3, 8};
    auto make_weights = [&](bool folding_disabled) {
        auto packed = v0::Constant::create(ov::element::u4, ov::Shape{groups, width, channels}, {1});
        auto converted = std::make_shared<v0::Convert>(packed, ov::element::f16);
        if (folding_disabled) {
            ov::pass::disable_constant_folding(converted);
        }
        auto scales = v0::Constant::create(ov::element::f16, ov::Shape{groups, width, 1}, {0.125f});
        auto scaled = std::make_shared<v1::Multiply>(converted, scales);
        return std::make_shared<v0::Convert>(scaled, f32);
    };
    {
        auto activations = std::make_shared<v0::Parameter>(f32, ov::Shape{static_cast<size_t>(ends.back()), channels});
        auto offsets = v0::Constant::create(i32, ov::Shape{groups}, ends);
        auto grouped = std::make_shared<v17::GroupedMatMul>(activations, make_weights(true), offsets);
        model = std::make_shared<ov::Model>(ov::OutputVector{grouped}, ov::ParameterVector{activations});
        manager.register_pass<ov::pass::ConvertGroupedMatMulToMatMul>();
    }
    {
        auto activations = std::make_shared<v0::Parameter>(f32, ov::Shape{static_cast<size_t>(ends.back()), channels});
        auto offsets = v0::Constant::create(i32, ov::Shape{groups}, ends);
        auto expected = ref_2dx3d_decomposition(activations, make_weights(false), offsets, groups);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{expected}, ov::ParameterVector{activations});
    }
}

TEST(TransformationTests, ConvertGroupedMatMulToMatMul_2Dx3D_SplitWeightsShareOriginalData) {
    constexpr size_t groups = 3, width = 8, channels = 256;
    const std::vector<int32_t> ends{2, 5, 8};
    std::vector<uint8_t> packed_values(groups * width * channels / 2);
    for (size_t i = 0; i < packed_values.size(); ++i) {
        packed_values[i] = static_cast<uint8_t>(i % 251);
    }
    auto packed =
        std::make_shared<v0::Constant>(ov::element::u4, ov::Shape{groups, width, channels}, packed_values.data());
    packed->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(packed->get_byte_size(), 0, ov::element::u4);
    auto converted = std::make_shared<v0::Convert>(packed, ov::element::f16);
    auto scales = v0::Constant::create(ov::element::f16, ov::Shape{groups, width, 1}, {0.125f});
    auto scaled = std::make_shared<v1::Multiply>(converted, scales);
    auto weights = std::make_shared<v0::Convert>(scaled, f32);
    auto activations = std::make_shared<v0::Parameter>(f32, ov::Shape{static_cast<size_t>(ends.back()), channels});
    auto offsets = v0::Constant::create(i32, ov::Shape{groups}, ends);
    auto grouped = std::make_shared<v17::GroupedMatMul>(activations, weights, offsets);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{grouped}, ov::ParameterVector{activations});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertGroupedMatMulToMatMul>();
    manager.run_passes(model);

    const auto* original = static_cast<const uint8_t*>(packed->get_data_ptr());
    const size_t expert_bytes = width * channels / 2;
    std::vector<const uint8_t*> expert_ptrs;
    for (const auto& op : model->get_ops()) {
        auto constant = ov::as_type_ptr<v0::Constant>(op);
        if (constant && constant->get_element_type() == ov::element::u4) {
            expert_ptrs.push_back(static_cast<const uint8_t*>(constant->get_data_ptr()));
            EXPECT_EQ(constant->get_rt_info().count(ov::WeightlessCacheAttribute::get_type_info_static()), 0);
        }
    }
    ASSERT_EQ(expert_ptrs.size(), groups);
    std::sort(expert_ptrs.begin(), expert_ptrs.end());
    for (size_t g = 0; g < groups; ++g) {
        EXPECT_EQ(expert_ptrs[g], original + g * expert_bytes);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QuantizedExperts,
    ConvertGroupedMatMulToMatMulCompressed,
    testing::Combine(testing::Values(ov::element::u4, ov::element::i4, ov::element::u8, ov::element::i8),
                     testing::Values(std::vector<int32_t>{3},
                                     std::vector<int32_t>{1, 3},
                                     std::vector<int32_t>{1, 3, 6, 10},
                                     std::vector<int32_t>{0, 2, 2, 10}),
                     testing::Bool()));

TEST_F(TransformationTestsF, ConvertGroupedMatMulToMatMul_OnlyCompressedSplit_SkipsUncompressedWeights) {
    constexpr size_t groups = 2, width = 8, channels = 256;
    const std::vector<int32_t> ends{3, 8};
    auto weights = v0::Constant::create(f32, ov::Shape{groups, width, channels}, {0.5f});
    auto activations = std::make_shared<v0::Parameter>(f32, ov::Shape{static_cast<size_t>(ends.back()), channels});
    auto offsets = v0::Constant::create(i32, ov::Shape{groups}, ends);
    auto grouped = std::make_shared<v17::GroupedMatMul>(activations, weights, offsets);
    model = std::make_shared<ov::Model>(ov::OutputVector{grouped}, ov::ParameterVector{activations});
    manager.register_pass<ov::pass::ConvertGroupedMatMulToMatMul>(true);
    // No model_ref: uncompressed weights are not split, so the GroupedMatMul is left for the regular decomposition.
}
