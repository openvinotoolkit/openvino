// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/result.hpp"
#include "intel_gpu/op/gemm.hpp"

#include "plugin/transformations/transpose_fusion.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, TransposeMatmulFusion1) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_a, input_b);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input_a, input_b});
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 1, 2, 3};
        std::vector<int64_t> order_b = {0, 1, 2, 3};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
                                                              input_b,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gemm}, ov::ParameterVector{input_a, input_b});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransposeMatmulFusion2) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto matmul = std::make_shared<ov::op::v0::MatMul>(tranpose_a, input_b);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input_a, input_b});
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 2, 1, 3};
        std::vector<int64_t> order_b = {0, 1, 2, 3};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
                                                              input_b,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gemm}, ov::ParameterVector{input_a, input_b});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransposeMatmulFusion3) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(tranpose_a, tranpose_b);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input_a, input_b});
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 2, 1, 3};
        std::vector<int64_t> order_b = {0, 1, 3, 2};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
                                                              input_b,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gemm}, ov::ParameterVector{input_a, input_b});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransposeMatmulFusion4) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(tranpose_a, tranpose_b);
        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(matmul, tranpose_c_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{tranpose_c}, ov::ParameterVector{input_a, input_b});
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 2, 1, 3};
        std::vector<int64_t> order_b = {0, 2, 1, 3};
        std::vector<int64_t> order_c = {0, 2, 1, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
                                                              input_b,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gemm}, ov::ParameterVector{input_a, input_b});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransposeMatmulFusion5) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_a, input_b);
        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(matmul, tranpose_c_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{tranpose_c}, ov::ParameterVector{input_a, input_b});

        const auto supports_immad = false;
        manager.register_pass<TransposeFusion>(supports_immad);
    }
    {
        std::vector<int64_t> order_a = {0, 1, 2};
        std::vector<int64_t> order_b = {0, 1, 2};
        std::vector<int64_t> order_c = {0, 2, 1};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
                                                              input_b,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gemm}, ov::ParameterVector{input_a, input_b});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransposeMatmulFusion6) {
    auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(2));
    auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(2));
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input_a, input_b);
    auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(matmul, tranpose_c_const);

    model = std::make_shared<ov::Model>(ov::OutputVector{tranpose_c}, ov::ParameterVector{input_a, input_b});

    const auto supports_immad = false;
    manager.register_pass<TransposeFusion>(supports_immad);

    model_ref = model->clone();
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST_F(TransformationTestsF, TransposeMatmulFusion7) {
    auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 4});
    auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input_a, input_b);
    auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(matmul, tranpose_c_const);

    model = std::make_shared<ov::Model>(ov::OutputVector{tranpose_c}, ov::ParameterVector{input_a, input_b});

    const auto supports_immad = false;
    manager.register_pass<TransposeFusion>(supports_immad);

    model_ref = model->clone();
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST_F(TransformationTestsF, TransposeMatmulFusion8) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        // Unsupported transpose order
        const std::vector<int64_t> target_transpose_order = {0, 3, 1, 2};
        auto transpose_order_a = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{target_transpose_order.size()}, target_transpose_order);
        auto transpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, transpose_order_a);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose_a, input_b);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input_a, input_b});
        manager.register_pass<TransposeFusion>();

        model_ref = model->clone();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransposeMatmulFusion9) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 20, 30, 40});
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 40, 30, 20});
        // Unsupported transpose order
        const std::vector<int64_t> target_transpose_order = {0, 3, 1, 2};
        auto transpose_order_a = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{target_transpose_order.size()}, target_transpose_order);
        auto transpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, transpose_order_a);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose_a, input_b);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input_a, input_b});
        bool support_immad = false;
        manager.register_pass<TransposeFusion>(support_immad);

        model_ref = model->clone();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransposeMatmulFusion_Illegal_1) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 20});
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{20, 30});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_a, input_b);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input_a, input_b});
        manager.register_pass<TransposeFusion>();
    }
}

//Test case is written for the case where we have two outputs (Transpose and another OP) are connected to MatMul. In this case, TransposeMatmulTranspose Fusion cannot happen
TEST_F(TransformationTestsF, TransposeMatmulFusion10) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 4 }, { 0, 2, 1, 3 });
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 4 }, { 0, 2, 1, 3 });
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(tranpose_a, tranpose_b);
        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 4 }, { 0, 2, 1, 3 });
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(matmul, tranpose_c_const);
        auto Shape_Of = std::make_shared<ov::op::v3::ShapeOf>(matmul);

        model = std::make_shared<ov::Model>(ov::OutputVector{ tranpose_c->output(0), Shape_Of }, ov::ParameterVector{ input_a, input_b });
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = { 0, 2, 1, 3 };
        std::vector<int64_t> order_b = { 0, 2, 1, 3 };
        std::vector<int64_t> order_c = { 0, 1, 2, 3 };
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a,
            input_b,
            order_a,
            order_b,
            order_c,
            ov::element::dynamic);

        auto tranpose_d_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 4 }, { 0, 2, 1, 3 });
        auto tranpose_d = std::make_shared<ov::op::v1::Transpose>(gemm, tranpose_d_const);
        auto Shape_Of = std::make_shared<ov::op::v3::ShapeOf>(gemm);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{ tranpose_d, Shape_Of }, ov::ParameterVector{ input_a, input_b });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}
}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
