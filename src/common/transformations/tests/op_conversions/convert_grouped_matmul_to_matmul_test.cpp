// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_grouped_matmul_to_matmul.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
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
                                             int64_t num_groups) {
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
        auto b_g = std::make_shared<v0::Squeeze>(b_split->output(g), axis_1d);
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
