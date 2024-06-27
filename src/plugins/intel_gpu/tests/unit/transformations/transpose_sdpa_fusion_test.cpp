// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "intel_gpu/op/sdpa.hpp"

#include "plugin/transformations/transpose_fusion.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, TranposeSDPAFusion1) {
    const bool is_causal = true;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(input_a, input_b, input_c, is_causal);

        model = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_a, input_b, input_c });
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 1, 2, 3};
        std::vector<int64_t> order_b = {0, 1, 2, 3};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        std::vector<int64_t> order_output = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, input_b, input_c}, is_causal, order_a, order_b, order_c, order_output, ov::element::undefined );

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_a, input_b, input_c });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransformationTestsF) {
    const bool is_causal = true;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(tranpose_a, input_b, input_c, is_causal);

        model = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_a, input_b, input_c });
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 2, 1, 3};
        std::vector<int64_t> order_b = {0, 1, 2, 3};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        std::vector<int64_t> order_output = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, input_b, input_c}, is_causal, order_a, order_b, order_c, order_output, ov::element::undefined);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_a, input_b, input_c });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TranposeSDPAFusion3) {
    const bool is_causal = false;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 2, 0, 3});
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(tranpose_a, tranpose_b, input_c, is_causal);

        model = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_a, input_b, input_c });
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 2, 1, 3};
        std::vector<int64_t> order_b = {1, 2, 0, 3};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        std::vector<int64_t> order_output = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, input_b, input_c}, is_causal, order_a, order_b, order_c, order_output, ov::element::undefined);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_a, input_b, input_c });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TranposeSDPAFusion4) {
    const bool is_causal = false;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(input_c, tranpose_c_const);

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(tranpose_a, tranpose_b, tranpose_c, is_causal);

        model = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_a, input_b, input_c });
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 2, 1, 3};
        std::vector<int64_t> order_b = {0, 2, 1, 3};
        std::vector<int64_t> order_c = {0, 2, 1, 3};
        std::vector<int64_t> order_output = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, input_b, input_c}, is_causal, order_a, order_b, order_c, order_output, ov::element::undefined);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_a, input_b, input_c });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TranposeSDPAFusion5) {
    const bool is_causal = false;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {3, 2, 1, 0});
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(input_c, tranpose_c_const);

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(tranpose_a, tranpose_b, tranpose_c, is_causal);

        model = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_a, input_b, input_c });
        manager.register_pass<TransposeFusion>();
    }
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {3, 2, 1, 0});
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(input_c, tranpose_c_const);

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(tranpose_a, tranpose_b, tranpose_c, is_causal);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ sdpa }, ov::ParameterVector{ input_a, input_b, input_c });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
