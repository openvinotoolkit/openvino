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
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/pass/manager.hpp"

namespace v0 = ov::op::v0;
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
    ov::Output<ov::Node> offsets_i32 = offsets;
    if (offsets.get_element_type() != i32) {
        offsets_i32 = std::make_shared<v0::Convert>(offsets, i32);
    }

    auto step = v0::Constant::create(i32, ov::Shape{1}, {1});
    auto slice_axis = v0::Constant::create(i32, ov::Shape{1}, {0});
    auto gather_axis = v0::Constant::create(i32, ov::Shape{}, {0});

    ov::OutputVector group_outputs;
    ov::Output<ov::Node> start = v0::Constant::create(i32, ov::Shape{1}, {0});
    for (int64_t g = 0; g < num_groups; ++g) {
        auto end_index = v0::Constant::create(i32, ov::Shape{1}, {g});
        ov::Output<ov::Node> end = std::make_shared<v8::Gather>(offsets_i32, end_index, gather_axis);
        auto a_g = std::make_shared<v8::Slice>(mat_a, start, end, step, slice_axis);
        auto b_index = v0::Constant::create(i32, ov::Shape{}, {g});
        auto b_g = std::make_shared<v8::Gather>(mat_b, b_index, gather_axis);
        auto mm = std::make_shared<v0::MatMul>(a_g, b_g, /*transpose_a=*/false, /*transpose_b=*/true);
        group_outputs.push_back(mm);
        start = end;
    }
    return std::make_shared<v0::Concat>(group_outputs, 0);
}

}  // namespace

// 3Dx3D: A:[G,M,K] B:[G,N,K] -> MatMul(A, B, transpose_b=true)
TEST_F(TransformationTestsF, ConvertGroupedMatMulToMatMul_3Dx3D) {
    constexpr size_t G = 2, M = 4, K = 8, N = 16;
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

// 2Dx3D with i32 offsets: no Convert is inserted
TEST_F(TransformationTestsF, ConvertGroupedMatMulToMatMul_2Dx3D_i32Offsets) {
    constexpr size_t T = 6, K = 8, G = 2, N = 16;
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);

    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{G, N, K});
        auto offsets = v0::Constant::create(i32, ov::Shape{G}, {3, 6});
        auto gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b, offsets);
        model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{mat_a, mat_b});

        manager.register_pass<ov::pass::ConvertGroupedMatMulToMatMul>();
    }
    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{G, N, K});
        auto offsets = v0::Constant::create(i32, ov::Shape{G}, {3, 6});
        auto out = ref_2dx3d_decomposition(mat_a, mat_b, offsets, G);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{mat_a, mat_b});
    }
}

// 2Dx3D with i64 offsets: a Convert to the i32 index type is inserted
TEST_F(TransformationTestsF, ConvertGroupedMatMulToMatMul_2Dx3D_i64Offsets) {
    constexpr size_t T = 6, K = 8, G = 2, N = 16;
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);

    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{G, N, K});
        auto offsets = v0::Constant::create(i64, ov::Shape{G}, {3, 6});
        auto gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b, offsets);
        model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{mat_a, mat_b});

        manager.register_pass<ov::pass::ConvertGroupedMatMulToMatMul>();
    }
    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{G, N, K});
        auto offsets = v0::Constant::create(i64, ov::Shape{G}, {3, 6});
        auto out = ref_2dx3d_decomposition(mat_a, mat_b, offsets, G);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{mat_a, mat_b});
    }
}

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

// -----------------------------------------------------------------------------
// Accuracy coverage for the exact shapes that fail on the GPU plugin's native
// GroupedMatMul lowering (batched fully_connected collapses the group dim,
// producing [G, M, G*N]). The decomposition below is verified numerically on
// the template plugin (ACCURACY mode), proving the transformation itself is
// correct for these shapes. Shapes use the op layout mat_b:[G, N, K].
// -----------------------------------------------------------------------------

// {G, M, K, N}
using GmmShape3D = std::tuple<size_t, size_t, size_t, size_t>;

class ConvertGroupedMatMulToMatMul3Dx3DAccuracy : public TransformationTestsF,
                                                  public testing::WithParamInterface<GmmShape3D> {};

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

INSTANTIATE_TEST_SUITE_P(GpuFailingShapes,
                         ConvertGroupedMatMulToMatMul3Dx3DAccuracy,
                         testing::Values(GmmShape3D{1, 8, 16, 8},   // passes on GPU (G==1)
                                         GmmShape3D{2, 3, 8, 16},   // GPU: ov=[2,3,32]
                                         GmmShape3D{4, 1, 8, 8},    // GPU: ov=[4,1,32]
                                         GmmShape3D{3, 7, 16, 8}));  // GPU: ov=[3,7,24]

// {T, K, N, cumulative offsets}
using GmmShape2D = std::tuple<size_t, size_t, size_t, std::vector<int32_t>>;

class ConvertGroupedMatMulToMatMul2Dx3DAccuracy : public TransformationTestsF,
                                                  public testing::WithParamInterface<GmmShape2D> {};

TEST_P(ConvertGroupedMatMulToMatMul2Dx3DAccuracy, MatchesReference) {
    const auto [T, K, N, offsets_vec] = GetParam();
    const auto G = static_cast<int64_t>(offsets_vec.size());
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);

    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{offsets_vec.size(), N, K});
        auto offsets = v0::Constant::create(i32, ov::Shape{offsets_vec.size()}, offsets_vec);
        auto gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b, offsets);
        model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{mat_a, mat_b});

        manager.register_pass<ov::pass::ConvertGroupedMatMulToMatMul>();
    }
    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{offsets_vec.size(), N, K});
        auto offsets = v0::Constant::create(i32, ov::Shape{offsets_vec.size()}, offsets_vec);
        auto out = ref_2dx3d_decomposition(mat_a, mat_b, offsets, G);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{mat_a, mat_b});
    }
}

INSTANTIATE_TEST_SUITE_P(GpuFailingShapes,
                         ConvertGroupedMatMulToMatMul2Dx3DAccuracy,
                         testing::Values(GmmShape2D{8, 8, 16, {8}},
                                         GmmShape2D{16, 16, 8, {8, 16}},
                                         GmmShape2D{24, 8, 8, {8, 16, 24}}));
