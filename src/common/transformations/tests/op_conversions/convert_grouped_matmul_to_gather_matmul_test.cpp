// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_grouped_matmul_to_gather_matmul.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/search_sorted.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/gather_matmul.hpp"

namespace v0 = ov::op::v0;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;
namespace v15 = ov::op::v15;
namespace v17 = ov::op::v17;

using GatherMatmul = ov::op::internal::GatherMatmul;

namespace {

constexpr auto f32 = ov::element::f32;
constexpr auto i32 = ov::element::i32;
constexpr auto i64 = ov::element::i64;

ov::Output<ov::Node> ref_3dx3d_indices(const ov::Output<ov::Node>& mat_a) {
    auto zero = v0::Constant::create(i32, ov::Shape{}, {0});
    auto shape_a = std::make_shared<v3::ShapeOf>(mat_a, i32);
    auto mg_idx = v0::Constant::create(i32, ov::Shape{2}, {1, 0});
    auto target_shape = std::make_shared<v8::Gather>(shape_a, mg_idx, zero);
    auto g_scalar = std::make_shared<v8::Gather>(shape_a, zero, zero);
    auto one = v0::Constant::create(i32, ov::Shape{}, {1});
    auto range = std::make_shared<v4::Range>(zero, g_scalar, one, i32);
    return std::make_shared<v3::Broadcast>(range, target_shape);
}

ov::Output<ov::Node> ref_2dx3d_indices(const ov::Output<ov::Node>& mat_a, const ov::Output<ov::Node>& offsets) {
    auto zero = v0::Constant::create(i32, ov::Shape{}, {0});

    ov::Output<ov::Node> offsets_i32 = offsets;
    if (offsets.get_element_type() != i32) {
        offsets_i32 = std::make_shared<v0::Convert>(offsets, i32);
    }

    auto shape_a = std::make_shared<v3::ShapeOf>(mat_a, i32);
    auto t_scalar = std::make_shared<v8::Gather>(shape_a, zero, zero);
    auto one = v0::Constant::create(i32, ov::Shape{}, {1});
    auto positions = std::make_shared<v4::Range>(zero, t_scalar, one, i32);
    auto idx_1d = std::make_shared<v15::SearchSorted>(offsets_i32, positions, /*right_mode=*/true, i32);
    auto unsqueeze_axis = v0::Constant::create(i32, ov::Shape{1}, {-1});
    return std::make_shared<v0::Unsqueeze>(idx_1d, unsqueeze_axis);
}

}  // namespace

// 3Dx3D: A:[G,M,K] B:[G,N,K] -> GatherMatmul(A, B, indices)
TEST_F(TransformationTestsF, ConvertGroupedMatMulToGatherMatmul_3Dx3D) {
    constexpr size_t G = 2, M = 4, K = 8, N = 16;

    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{G, M, K});
        auto mat_b = v0::Constant::create(f32, ov::Shape{G, N, K}, {1.0f});
        auto gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b);
        model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{mat_a});

        manager.register_pass<ov::pass::ConvertGroupedMatMulToGatherMatmul>();
    }
    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{G, M, K});
        auto mat_b = v0::Constant::create(f32, ov::Shape{G, N, K}, {1.0f});
        auto indices = ref_3dx3d_indices(mat_a);
        auto gm = std::make_shared<GatherMatmul>(mat_a, mat_b, indices);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gm}, ov::ParameterVector{mat_a});
    }
}

// 2Dx3D with i32 offsets: no Convert is inserted
TEST_F(TransformationTestsF, ConvertGroupedMatMulToGatherMatmul_2Dx3D_i32Offsets) {
    constexpr size_t T = 6, K = 8, G = 2, N = 16;

    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = v0::Constant::create(f32, ov::Shape{G, N, K}, {1.0f});
        auto offsets = v0::Constant::create(i32, ov::Shape{G}, {3, 6});
        auto gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b, offsets);
        model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{mat_a});

        manager.register_pass<ov::pass::ConvertGroupedMatMulToGatherMatmul>();
    }
    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = v0::Constant::create(f32, ov::Shape{G, N, K}, {1.0f});
        auto offsets = v0::Constant::create(i32, ov::Shape{G}, {3, 6});

        auto a_unsq_axis = v0::Constant::create(i32, ov::Shape{1}, {0});
        auto a_3d = std::make_shared<v0::Unsqueeze>(mat_a, a_unsq_axis);
        auto indices = ref_2dx3d_indices(mat_a, offsets);
        auto gm = std::make_shared<GatherMatmul>(a_3d, mat_b, indices);
        auto squeeze_axis = v0::Constant::create(i32, ov::Shape{1}, {0});
        auto out = std::make_shared<v0::Squeeze>(gm, squeeze_axis);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{mat_a});
    }
}

// 2Dx3D with i64 offsets: a Convert to i32 is inserted before SearchSorted
TEST_F(TransformationTestsF, ConvertGroupedMatMulToGatherMatmul_2Dx3D_i64OffsetsInsertsConvert) {
    constexpr size_t T = 6, K = 8, G = 2, N = 16;

    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = v0::Constant::create(f32, ov::Shape{G, N, K}, {1.0f});
        auto offsets = v0::Constant::create(i64, ov::Shape{G}, {3, 6});
        auto gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b, offsets);
        model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{mat_a});

        manager.register_pass<ov::pass::ConvertGroupedMatMulToGatherMatmul>();
    }
    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{T, K});
        auto mat_b = v0::Constant::create(f32, ov::Shape{G, N, K}, {1.0f});
        auto offsets = v0::Constant::create(i64, ov::Shape{G}, {3, 6});

        auto a_unsq_axis = v0::Constant::create(i32, ov::Shape{1}, {0});
        auto a_3d = std::make_shared<v0::Unsqueeze>(mat_a, a_unsq_axis);
        auto indices = ref_2dx3d_indices(mat_a, offsets);  // inserts Convert internally
        auto gm = std::make_shared<GatherMatmul>(a_3d, mat_b, indices);
        auto squeeze_axis = v0::Constant::create(i32, ov::Shape{1}, {0});
        auto out = std::make_shared<v0::Squeeze>(gm, squeeze_axis);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{mat_a});
    }
}

// Input B is not a Constant -> no conversion
TEST_F(TransformationTestsF, ConvertGroupedMatMulToGatherMatmul_NonConstantWeightsNoChange) {
    constexpr size_t G = 2, M = 4, K = 8, N = 16;

    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{G, M, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{G, N, K});
        auto gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b);
        model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{mat_a, mat_b});

        manager.register_pass<ov::pass::ConvertGroupedMatMulToGatherMatmul>();
    }
    {
        auto mat_a = std::make_shared<v0::Parameter>(f32, ov::Shape{G, M, K});
        auto mat_b = std::make_shared<v0::Parameter>(f32, ov::Shape{G, N, K});
        auto gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{mat_a, mat_b});
    }
}
