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
#include "openvino/op/multiply.hpp"
#include "openvino/op/convert.hpp"
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
        auto weight_fused = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1664, 4096});
        auto bias1 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_fused = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1664, 32});
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

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_parameter_weights_no_fusion) {
    // Weights-as-inputs / share_weights rewrites weight Constants into Parameters. The fused-weight
    // Concat would be unfoldable and reach GPU program build (no i4/u4 concat kernel). Horizontal
    // fusion must NOT fire in this case, so model_ref is identical to model (3 separate FCs).
    std::vector<int64_t> pattern = {7, -1};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 7, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{1024, 4096});
        weight1->set_friendly_name("weight1_1");
        auto weight2 = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{512, 4096});
        weight2->set_friendly_name("weight1_2");
        auto weight3 = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{128, 4096});
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
        model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input, weight1, weight2, weight3});
        manager.register_pass<FullyConnectedHorizontalFusion>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 7, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{1024, 4096});
        weight1->set_friendly_name("weight1_1");
        auto weight2 = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{512, 4096});
        weight2->set_friendly_name("weight1_2");
        auto weight3 = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{128, 4096});
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
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input, weight1, weight2, weight3});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_convert_parameter_weights_no_fusion) {
    // Convert(Parameter) weight form must also block horizontal fusion (is_constant returns false
    // because the Convert input is a Parameter, not a Constant).
    std::vector<int64_t> pattern = {7, -1};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 7, 4096});
        auto weight1_p = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{1024, 4096});
        auto weight2_p = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{512, 4096});
        auto weight3_p = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{128, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Convert>(weight1_p, ov::element::f16);
        auto weight2 = std::make_shared<ov::op::v0::Convert>(weight2_p, ov::element::f16);
        auto weight3 = std::make_shared<ov::op::v0::Convert>(weight3_p, ov::element::f16);
        auto bias1 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias2 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias3 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{512, 32});
        auto scale3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{128, 32});
        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight1, bias1, scale1);
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight2, bias2, scale2);
        auto fc3 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight3, bias3, scale3);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(fc1, reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(fc2, reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(fc3, reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input, weight1_p, weight2_p, weight3_p});
        manager.register_pass<FullyConnectedHorizontalFusion>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 7, 4096});
        auto weight1_p = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{1024, 4096});
        auto weight2_p = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{512, 4096});
        auto weight3_p = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{128, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Convert>(weight1_p, ov::element::f16);
        auto weight2 = std::make_shared<ov::op::v0::Convert>(weight2_p, ov::element::f16);
        auto weight3 = std::make_shared<ov::op::v0::Convert>(weight3_p, ov::element::f16);
        auto bias1 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias2 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias3 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{512, 32});
        auto scale3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{128, 32});
        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight1, bias1, scale1);
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight2, bias2, scale2);
        auto fc3 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight3, bias3, scale3);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(fc1, reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(fc2, reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(fc3, reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input, weight1_p, weight2_p, weight3_p});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_bias_no_zp) {
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
        auto weight_fused = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1664, 4096});
        auto bias_fused = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1664});
        auto scale_fused = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1664, 32});
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
        auto weight_fused = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1664, 4096});
        auto bias_fused = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1664});
        auto scale_fused = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1664, 32});
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
        auto weight_fused = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1664, 4096});
        auto bias_fused = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1664});
        auto scale_fused = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1664, 32});
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
TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_transpose_b_false) {
    // transpose_b=false => weights are in [K, N] layout (opposite of the default [N, K]).
    // Horizontal fusion must concat along the N axis (1) and use the N dims as split sizes,
    // otherwise the fused FC shape-inference throws (contraction dim mismatch).
    std::vector<int64_t> pattern = {7, -1};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 7, 4096});
        // [K, N] weights: K=4096, N=1024 / 512 / 128
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{4096, 1024});
        weight1->set_friendly_name("weight1_1");
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{4096, 512});
        weight2->set_friendly_name("weight1_2");
        auto weight3 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{4096, 128});
        weight3->set_friendly_name("weight1_3");
        auto bias1 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias2 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias3 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{512, 32});
        auto scale3 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{128, 32});
        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight1, bias1, scale1, ov::element::dynamic, false);
        fc1->set_friendly_name("fc1");
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight2, bias2, scale2, ov::element::dynamic, false);
        auto fc3 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight3, bias3, scale3, ov::element::dynamic, false);
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
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 7, 4096});
        // [K, N] weights
        auto weight_fused = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{4096, 1664});
        auto bias1 = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_fused = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1664, 32});
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_fused, bias1, scale_fused, ov::element::dynamic, false);
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

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_u3_weights_byte_packed) {
    // u3 weights are byte-packed, so the fused weight must preserve the original packed data
    const std::vector<int64_t> pattern = {7, -1};
    auto make_weight = [](const ov::Shape& shape, const std::vector<uint8_t>& bytes) {
        return std::make_shared<ov::op::v0::Constant>(ov::element::u3, shape, bytes.data());
    };
    auto make_scale = [](size_t output_channels) {
        return std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{output_channels, 1});
    };
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 7, 8});
        auto weight1 = make_weight(ov::Shape{2, 8}, {0x11, 0x22, 0x33, 0x44, 0x55, 0x66});
        auto weight2 = make_weight(ov::Shape{3, 8}, {0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF});
        auto weight3 = make_weight(ov::Shape{1, 8}, {0x01, 0x02, 0x03});
        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                 weight1,
                                                                                 std::make_shared<ov::intel_gpu::op::Placeholder>(),
                                                                                 make_scale(2));
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                 weight2,
                                                                                 std::make_shared<ov::intel_gpu::op::Placeholder>(),
                                                                                 make_scale(3));
        auto fc3 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                 weight3,
                                                                                 std::make_shared<ov::intel_gpu::op::Placeholder>(),
                                                                                 make_scale(1));
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
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 7, 8});
        auto fused_weight = make_weight(ov::Shape{6, 8},
                                        {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
                                         0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x01, 0x02, 0x03});
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                      fused_weight,
                                                                                      std::make_shared<ov::intel_gpu::op::Placeholder>(),
                                                                                      make_scale(6));
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {2, 3, 1});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fc_fused, axis_const, split_const);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(split->output(0), reshape_pattern, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(split->output(1), reshape_pattern, true);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(split->output(2), reshape_pattern, true);
        auto result1 = std::make_shared<ov::op::v0::Result>(reshape1);
        auto result2 = std::make_shared<ov::op::v0::Result>(reshape2);
        auto result3 = std::make_shared<ov::op::v0::Result>(reshape3);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result1, result2, result3}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_u3_misaligned_no_fusion) {
    // With K=4 each u3 row occupies 12 bits, so Concat along N (or along the zp's first axis) is not byte aligned and
    // Concat::evaluate fails. The fused constant can't be folded, hence the transformation must bail out and leave
    // the FCs untouched. Two independent groups of 3 FCs cover different failure points:
    //  1. u3 weights: the weights folding fails.
    //  2. u4 weights with u3 zero points and bias Adds: the zero points folding fails after the biases are collected.
    //     The bias Adds must not be bypassed, as the pass has to leave the model intact when it returns false.
    auto make_model = []() {
        ov::ResultVector results;
        ov::ParameterVector params;
        {
            auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 7, 4});
            params.push_back(input);
            for (size_t n : {2, 3, 1}) {
                auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::u3, ov::Shape{n, 4});
                auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{n, 1});
                auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                        weight,
                                                                                        std::make_shared<ov::intel_gpu::op::Placeholder>(),
                                                                                        scale);
                results.push_back(std::make_shared<ov::op::v0::Result>(fc));
            }
        }
        {
            auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 7, 4096});
            params.push_back(input);
            for (size_t n : {1024, 512, 128}) {
                auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{n, 4096});
                auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{n, 32});
                auto zp = std::make_shared<ov::op::v0::Constant>(ov::element::u3, ov::Shape{1, 1, 4});
                auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                        weight,
                                                                                        std::make_shared<ov::intel_gpu::op::Placeholder>(),
                                                                                        scale,
                                                                                        zp);
                auto add_input = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, n});
                auto add = std::make_shared<ov::op::v1::Add>(fc, add_input);
                results.push_back(std::make_shared<ov::op::v0::Result>(add));
            }
        }
        return std::make_shared<ov::Model>(results, params);
    };
    model = make_model();
    model_ref = make_model();
    manager.register_pass<FullyConnectedHorizontalFusion>();
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov