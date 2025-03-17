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
#include "openvino/op/result.hpp"
#include "intel_gpu/op/gemm.hpp"

#include "plugin/transformations/transpose_fusion.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, TranposeMatmulFusion1) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_a, input_b);

        model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input_a, input_b });
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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, input_b });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TranposeMatmulFusion2) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto matmul = std::make_shared<ov::op::v0::MatMul>(tranpose_a, input_b);

        model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input_a, input_b });
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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, input_b });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TranposeMatmulFusion3) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(tranpose_a, tranpose_b);

        model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input_a, input_b });
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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, input_b });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TranposeMatmulFusion4) {
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

        model = std::make_shared<ov::Model>(ov::NodeVector{ tranpose_c }, ov::ParameterVector{ input_a, input_b });
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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, input_b });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TranposeMatmulFusion5) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_a, input_b);
        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(matmul, tranpose_c_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{ tranpose_c }, ov::ParameterVector{ input_a, input_b });

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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, input_b });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TranposeMatmulFusion6) {
    auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(2));
    auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(2));
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input_a, input_b);
    auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(matmul, tranpose_c_const);

    model = std::make_shared<ov::Model>(ov::NodeVector{ tranpose_c }, ov::ParameterVector{ input_a, input_b });

    const auto supports_immad = false;
    manager.register_pass<TransposeFusion>(supports_immad);

    model_ref = model->clone();
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST_F(TransformationTestsF, TranposeMatmulFusion7) {
    auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 4});
    auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input_a, input_b);
    auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(matmul, tranpose_c_const);

    model = std::make_shared<ov::Model>(ov::NodeVector{ tranpose_c }, ov::ParameterVector{ input_a, input_b });

    const auto supports_immad = false;
    manager.register_pass<TransposeFusion>(supports_immad);

    model_ref = model->clone();
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST_F(TransformationTestsF, TranposeMatmulFusion_Illegal_1) {
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 20});
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{20, 30});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_a, input_b);

        model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input_a, input_b });
        manager.register_pass<TransposeFusion>();
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
