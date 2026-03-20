// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"

#include <string>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/manager.hpp"

#include <transformations/utils/utils.hpp>
#include "plugin/transformations/fc_horizontal_fusion.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_no_bias_no_zp) {
    std::vector<int64_t> pattern = {7, -1};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 7, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        weight1->set_friendly_name("weight1_1");
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{512, 4096});
        weight2->set_friendly_name("weight1_2");
        auto weight3 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{128, 4096});
        weight3->set_friendly_name("weight1_3");
        auto bias1 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias2 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias3 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{512, 32});
        auto scale3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{128, 32});
        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight1, bias1, scale1);
        fc1->set_friendly_name("fc1");
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight2, bias2, scale2);
        auto fc3 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight3, bias3, scale3);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(fc1, reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(fc2, reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(fc3, reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input});
        manager.register_pass<FullyConnectedHorizontalFusion>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 7, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        weight1->set_friendly_name("weight2_1");
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{512, 4096});
        weight2->set_friendly_name("weight2_2");
        auto weight3 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{128, 4096});
        weight3->set_friendly_name("weight2_3");
        auto weights = ov::OutputVector{weight1, weight2, weight3};
        auto weight_fused = std::make_shared<ov::op::v0::Concat>(weights, 0);
        auto bias1 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{512, 32});
        auto scale3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{128, 32});
        auto scales = ov::OutputVector{scale1, scale2, scale3};
        auto scale_fused = std::make_shared<ov::op::v0::Concat>(scales, 0);
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_fused, bias1, scale_fused);
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {fc_fused->get_output_partial_shape(0).size() - 1});
        std::vector<int64_t> orig_n_sizes = {1024, 512, 128};
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, orig_n_sizes);
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fc_fused, axis_const, split_const);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(split->output(0), reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(split->output(1), reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(split->output(2), reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_bias_zp) {
    std::vector<int64_t> pattern = {7, -1};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 7, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        weight1->set_friendly_name("weight1_1");
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{512, 4096});
        weight2->set_friendly_name("weight1_2");
        auto weight3 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{128, 4096});
        weight3->set_friendly_name("weight1_3");

        auto bias1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1024});
        auto bias2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 512});
        auto bias3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 128});
 
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{512, 32});
        auto scale3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{128, 32});
        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight1, bias1, scale1);
        fc1->set_friendly_name("fc1");
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight2, bias2, scale2);
        auto fc3 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight3, bias3, scale3);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(fc1, reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(fc2, reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(fc3, reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input});
        manager.register_pass<FullyConnectedHorizontalFusion>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 7, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        weight1->set_friendly_name("weight2_1");
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{512, 4096});
        weight2->set_friendly_name("weight2_2");
        auto weight3 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{128, 4096});
        weight3->set_friendly_name("weight2_3");
        auto weights = ov::OutputVector{weight1, weight2, weight3};
        auto weight_fused = std::make_shared<ov::op::v0::Concat>(weights, 0);
        auto bias1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1024});
        auto bias2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 512});
        auto bias3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 128});
        auto biases = ov::OutputVector{bias1, bias2, bias3};
        auto bias_fused = std::make_shared<ov::op::v0::Concat>(biases, 1);
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{512, 32});
        auto scale3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{128, 32});
        auto scales = ov::OutputVector{scale1, scale2, scale3};
        auto scale_fused = std::make_shared<ov::op::v0::Concat>(scales, 0);
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_fused, bias_fused, scale_fused);
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {fc_fused->get_output_partial_shape(0).size() - 1});
        std::vector<int64_t> orig_n_sizes = {1024, 512, 128};
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, orig_n_sizes);
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fc_fused, axis_const, split_const);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(split->output(0), reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(split->output(1), reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(split->output(2), reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_eltwise_bias_zp) {
    std::vector<int64_t> pattern = {7, -1};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 7, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        weight1->set_friendly_name("weight1_1");
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{512, 4096});
        weight2->set_friendly_name("weight1_2");
        auto weight3 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{128, 4096});
        weight3->set_friendly_name("weight1_3");

        auto bias1 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias2 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias3 = std::make_shared<ov::intel_gpu::op::Placeholder>();
 
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{512, 32});
        auto scale3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{128, 32});
        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight1, bias1, scale1);
        fc1->set_friendly_name("fc1");
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight2, bias2, scale2);
        auto fc3 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight3, bias3, scale3);

        auto add_input1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1024});
        auto add1 = std::make_shared<ov::op::v1::Add>(fc1, add_input1);

        auto add_input2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 512});
        auto add2 = std::make_shared<ov::op::v1::Add>(fc2, add_input2);

        auto add_input3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 128});
        auto add3 = std::make_shared<ov::op::v1::Add>(fc3, add_input3);

        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(add1, reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(add2, reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(add3, reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input});
        manager.register_pass<FullyConnectedHorizontalFusion>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 7, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        weight1->set_friendly_name("weight2_1");
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{512, 4096});
        weight2->set_friendly_name("weight2_2");
        auto weight3 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{128, 4096});
        weight3->set_friendly_name("weight2_3");
        auto weights = ov::OutputVector{weight1, weight2, weight3};
        auto weight_fused = std::make_shared<ov::op::v0::Concat>(weights, 0);
        auto bias1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1024});
        auto bias2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 512});
        auto bias3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 128});
        auto biases = ov::OutputVector{bias1, bias2, bias3};
        auto bias_fused = std::make_shared<ov::op::v0::Concat>(biases, 1);
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{512, 32});
        auto scale3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{128, 32});
        auto scales = ov::OutputVector{scale1, scale2, scale3};
        auto scale_fused = std::make_shared<ov::op::v0::Concat>(scales, 0);
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_fused, bias_fused, scale_fused);
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {fc_fused->get_output_partial_shape(0).size() - 1});
        std::vector<int64_t> orig_n_sizes = {1024, 512, 128};
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, orig_n_sizes);
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fc_fused, axis_const, split_const);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(split->output(0), reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(split->output(1), reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(split->output(2), reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_eltwise_bias_zp_scaling) {
    std::vector<int64_t> pattern = {7, -1};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 7, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        weight1->set_friendly_name("weight1_1");
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{512, 4096});
        weight2->set_friendly_name("weight1_2");
        auto weight3 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{128, 4096});
        weight3->set_friendly_name("weight1_3");

        auto bias1 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias2 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias3 = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{512, 32});
        auto scale3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{128, 32});
        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight1, bias1, scale1);
        fc1->set_friendly_name("fc1");
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight2, bias2, scale2);
        auto fc3 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight3, bias3, scale3);

        auto add_input1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1024});
        auto add1 = std::make_shared<ov::op::v1::Add>(fc1, add_input1);

        auto add_input2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 512});
        auto add2 = std::make_shared<ov::op::v1::Add>(fc2, add_input2);

        auto add_input3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 128});
        auto add3 = std::make_shared<ov::op::v1::Add>(fc3, add_input3);

        const std::vector<float> scale_factor = {8.f};
        auto mul_input1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, scale_factor);
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(add1, mul_input1);

        auto mul_input2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, scale_factor);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(add2, mul_input2);

        auto mul_input3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, scale_factor);
        auto mul3 = std::make_shared<ov::op::v1::Multiply>(add3, mul_input3);

        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(mul1, reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(mul2, reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(mul3, reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input});
        manager.register_pass<FullyConnectedHorizontalFusion>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 7, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        weight1->set_friendly_name("weight2_1");
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{512, 4096});
        weight2->set_friendly_name("weight2_2");
        auto weight3 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{128, 4096});
        weight3->set_friendly_name("weight2_3");
        auto weights = ov::OutputVector{weight1, weight2, weight3};
        auto weight_fused = std::make_shared<ov::op::v0::Concat>(weights, 0);
        auto bias1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1024});
        auto bias2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 512});
        auto bias3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 128});
        auto biases = ov::OutputVector{bias1, bias2, bias3};
        auto bias_fused = std::make_shared<ov::op::v0::Concat>(biases, 1);
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{512, 32});
        auto scale3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{128, 32});
        auto scales = ov::OutputVector{scale1, scale2, scale3};
        auto scale_fused = std::make_shared<ov::op::v0::Concat>(scales, 0);
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_fused, bias_fused, scale_fused);
        const std::vector<float> scale_factor = {8.f};
        auto mul_input = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, scale_factor);
        auto mul = std::make_shared<ov::op::v1::Multiply>(fc_fused, mul_input);
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {fc_fused->get_output_partial_shape(0).size() - 1});
        std::vector<int64_t> orig_n_sizes = {1024, 512, 128};
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, orig_n_sizes);
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(mul, axis_const, split_const);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(split->output(0), reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(split->output(1), reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(split->output(2), reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_skip_lora_consumers) {
    std::vector<int64_t> pattern = {7, -1};
    size_t hidden_size = 4096;
    size_t lora_rank = 128;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 7, static_cast<int64_t>(hidden_size)});
        auto weight_q = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{4096, hidden_size});
        auto weight_k = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, hidden_size});
        auto weight_v = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, hidden_size});
        auto bias_q = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias_k = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias_v = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_q = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{4096, 32});
        auto scale_k = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale_v = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto fc_q = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_q, bias_q, scale_q);
        auto fc_k = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_k, bias_k, scale_k);
        auto fc_v = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_v, bias_v, scale_v);

        // LoRA subgraph for Q: input -> MatMul(A) -> Multiply(alpha) -> MatMul(B) -> Add(FC_out, ...)
        auto lora_a_q = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{hidden_size, lora_rank});
        auto lora_matmul1_q = std::make_shared<ov::op::v0::MatMul>(input, lora_a_q, false, false);
        auto lora_alpha_q = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, lora_rank});
        auto lora_mul_q = std::make_shared<ov::op::v1::Multiply>(lora_matmul1_q, lora_alpha_q);
        auto lora_b_q = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{lora_rank, 4096});
        auto lora_matmul2_q = std::make_shared<ov::op::v0::MatMul>(lora_mul_q, lora_b_q, false, false);
        auto lora_add_q = std::make_shared<ov::op::v1::Add>(fc_q, lora_matmul2_q);

        // LoRA subgraph for K
        auto lora_a_k = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{hidden_size, lora_rank});
        auto lora_matmul1_k = std::make_shared<ov::op::v0::MatMul>(input, lora_a_k, false, false);
        auto lora_alpha_k = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, lora_rank});
        auto lora_mul_k = std::make_shared<ov::op::v1::Multiply>(lora_matmul1_k, lora_alpha_k);
        auto lora_b_k = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{lora_rank, 1024});
        auto lora_matmul2_k = std::make_shared<ov::op::v0::MatMul>(lora_mul_k, lora_b_k, false, false);
        auto lora_add_k = std::make_shared<ov::op::v1::Add>(fc_k, lora_matmul2_k);

        // LoRA subgraph for V
        auto lora_a_v = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{hidden_size, lora_rank});
        auto lora_matmul1_v = std::make_shared<ov::op::v0::MatMul>(input, lora_a_v, false, false);
        auto lora_alpha_v = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, lora_rank});
        auto lora_mul_v = std::make_shared<ov::op::v1::Multiply>(lora_matmul1_v, lora_alpha_v);
        auto lora_b_v = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{lora_rank, 1024});
        auto lora_matmul2_v = std::make_shared<ov::op::v0::MatMul>(lora_mul_v, lora_b_v, false, false);
        auto lora_add_v = std::make_shared<ov::op::v1::Add>(fc_v, lora_matmul2_v);

        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(lora_add_q, reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(lora_add_k, reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(lora_add_v, reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input});
        manager.register_pass<FullyConnectedHorizontalFusion>();
    }
    // model_ref is not set => expects no transformation (model unchanged)
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov