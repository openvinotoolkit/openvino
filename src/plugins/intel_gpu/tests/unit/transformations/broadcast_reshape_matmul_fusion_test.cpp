// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "intel_gpu/op/gemm.hpp"

#include "plugin/transformations/broadcast_reshape_matmul_fusion.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, BroadReshapeMatmulFusion1) {
    std::vector<int64_t> order_a = {0, 1, 2, 3};
    std::vector<int64_t> order_b = {1, 2, 3, 0};
    std::vector<int64_t> order_c = {0, 1, 2, 3};
    std::vector<int32_t> target_shape_b = {1, 1, 1, 16, 1};
    std::vector<int64_t> pattern_b = {0, 0, 32, 32};
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2, 1, 32});
        auto broadcast_b_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{5}, target_shape_b);
        auto broadcast_b = std::make_shared<ov::op::v3::Broadcast>(input_b, broadcast_b_const, ov::op::BroadcastType::BIDIRECTIONAL);
        auto reshape_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, pattern_b);
        auto reshape_b = std::make_shared<ov::op::v1::Reshape>(broadcast_b, reshape_b_const, true);
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a, reshape_b, order_a, order_b, order_c, ov::element::undefined);

        model = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, input_b });
        manager.register_pass<BroadcastReshapeMatmulFusion>();
    }
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2, 1, 32});
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a, input_b, std::vector<int32_t>{}, target_shape_b, std::vector<int64_t>{}, pattern_b, order_a, order_b, order_c, ov::element::undefined);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, input_b });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, BroadReshapeMatmulFusion2) {
    std::vector<int64_t> order_a = {0, 1, 2, 3};
    std::vector<int64_t> order_b = {1, 2, 3, 0};
    std::vector<int64_t> order_c = {0, 1, 2, 3};
    std::vector<int32_t> target_shape_b = {1, 1, 1, 16, 1};
    std::vector<int64_t> pattern_b = {0, 0, -1, 32};
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, 1, 32});
        auto broadcast_b_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{5}, target_shape_b);
        auto broadcast_b = std::make_shared<ov::op::v3::Broadcast>(input_b, broadcast_b_const, ov::op::BroadcastType::BIDIRECTIONAL);
        auto reshape_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, pattern_b);
        auto reshape_b = std::make_shared<ov::op::v1::Reshape>(broadcast_b, reshape_b_const, true);
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a, reshape_b, order_a, order_b, order_c, ov::element::undefined);

        model = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, input_b });
        manager.register_pass<BroadcastReshapeMatmulFusion>();
    }
    {
        model_ref = model->clone();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, BroadReshapeMatmulFusion3) {
    std::vector<int64_t> order_a = {0, 1, 2, 3};
    std::vector<int64_t> order_b = {0, 1, 2, 3};
    std::vector<int64_t> order_c = {0, 1, 2, 3};
    std::vector<int32_t> target_shape_b = {1, 1, 16, 1, 1};
    std::vector<int64_t> pattern_b = {0, 32, 32, 0};
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 2, 1, 32, -1});
        auto broadcast_b_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{5}, target_shape_b);
        auto broadcast_b = std::make_shared<ov::op::v3::Broadcast>(input_b, broadcast_b_const, ov::op::BroadcastType::BIDIRECTIONAL);
        auto reshape_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, pattern_b);
        auto reshape_b = std::make_shared<ov::op::v1::Reshape>(broadcast_b, reshape_b_const, true);
        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_a, reshape_b, order_a, order_b, order_c, ov::element::undefined);

        model = std::make_shared<ov::Model>(ov::NodeVector{ gemm }, ov::ParameterVector{ input_a, input_b });
        manager.register_pass<BroadcastReshapeMatmulFusion>();
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
