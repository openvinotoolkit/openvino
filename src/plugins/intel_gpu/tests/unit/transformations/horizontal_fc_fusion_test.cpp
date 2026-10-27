// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
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

namespace {
// Build a gguf_q4_1 Constant of logical shape [n, k] filled with a deterministic, position-dependent
// byte pattern (derived from `tag`), so the fused weight's raw bytes can be checked against a
// hand-built concatenation rather than just graph shape/attributes.
std::shared_ptr<ov::op::v0::Constant> make_gguf_q4_1_weight(size_t n, size_t k, uint8_t tag) {
    const auto& t = ov::element::gguf_q4_1;
    const size_t blocks_per_row = k / t.block_elem_count();
    const size_t byte_size = n * blocks_per_row * t.block_byte_size();
    std::vector<uint8_t> data(byte_size);
    for (size_t i = 0; i < byte_size; ++i) {
        data[i] = static_cast<uint8_t>(tag + i);
    }
    return std::make_shared<ov::op::v0::Constant>(t, ov::Shape{n, k}, data.data());
}

// Raw byte-concat of gguf_q4_1 weights along N, mirroring what
// FullyConnectedHorizontalFusion's dedicated GGUF fusion path is expected to build (never via
// ov::op::v0::Concat, since gguf_* element types report a bogus per-element byte size).
std::shared_ptr<ov::op::v0::Constant> concat_gguf_q4_1_weights_ref(
    const std::vector<std::shared_ptr<ov::op::v0::Constant>>& weights,
    size_t k) {
    const auto& t = ov::element::gguf_q4_1;
    size_t total_n = 0;
    size_t total_bytes = 0;
    for (const auto& w : weights) {
        total_n += w->get_shape()[0];
        total_bytes += w->get_byte_size();
    }
    std::vector<uint8_t> data(total_bytes);
    size_t offset = 0;
    for (const auto& w : weights) {
        std::memcpy(data.data() + offset, w->get_data_ptr(), w->get_byte_size());
        offset += w->get_byte_size();
    }
    return std::make_shared<ov::op::v0::Constant>(t, ov::Shape{total_n, k}, data.data());
}
}  // namespace

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_gguf_q4_1_qkv) {
    const size_t k = 4096;
    std::vector<int64_t> pattern = {7, -1};
    auto weight_q = make_gguf_q4_1_weight(4096, k, 0x10);
    auto weight_k = make_gguf_q4_1_weight(1024, k, 0x40);
    auto weight_v = make_gguf_q4_1_weight(1024, k, 0x80);
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 7, 4096});
        auto scale_q = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}, std::vector<float>{1.0f});
        auto scale_k = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}, std::vector<float>{1.0f});
        auto scale_v = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}, std::vector<float>{1.0f});
        auto bias_q = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias_k = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias_v = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto zp_q = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto zp_k = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto zp_v = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_q = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_q, bias_q, scale_q, zp_q);
        fc_q->set_friendly_name("fc_q");
        auto fc_k = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_k, bias_k, scale_k, zp_k);
        auto fc_v = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_v, bias_v, scale_v, zp_v);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape_q = std::make_shared<ov::op::v1::Reshape>(fc_q, reshape_pattern, true);
        auto reshape_k = std::make_shared<ov::op::v1::Reshape>(fc_k, reshape_pattern, true);
        auto reshape_v = std::make_shared<ov::op::v1::Reshape>(fc_v, reshape_pattern, true);
        auto result_q = std::make_shared<ov::op::v0::Result>(reshape_q);
        auto result_k = std::make_shared<ov::op::v0::Result>(reshape_k);
        auto result_v = std::make_shared<ov::op::v0::Result>(reshape_v);
        model = std::make_shared<ov::Model>(ov::ResultVector{result_q, result_k, result_v}, ov::ParameterVector{input});
        manager.register_pass<FullyConnectedHorizontalFusion>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 7, 4096});
        auto weight_fused = concat_gguf_q4_1_weights_ref({weight_q, weight_k, weight_v}, k);
        auto scale_q = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}, std::vector<float>{1.0f});
        auto scale_k = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}, std::vector<float>{1.0f});
        auto scale_v = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}, std::vector<float>{1.0f});
        auto scales = ov::OutputVector{scale_q, scale_k, scale_v};
        auto scale_fused = std::make_shared<ov::op::v0::Concat>(scales, 0);
        auto bias_fused = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto zp_fused = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_fused = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_fused, bias_fused, scale_fused, zp_fused);
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {fc_fused->get_output_partial_shape(0).size() - 1});
        std::vector<int64_t> orig_n_sizes = {4096, 1024, 1024};
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, orig_n_sizes);
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fc_fused, axis_const, split_const);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape_q = std::make_shared<ov::op::v1::Reshape>(split->output(0), reshape_pattern, true);
        auto reshape_k = std::make_shared<ov::op::v1::Reshape>(split->output(1), reshape_pattern, true);
        auto reshape_v = std::make_shared<ov::op::v1::Reshape>(split->output(2), reshape_pattern, true);
        auto result_q = std::make_shared<ov::op::v0::Result>(reshape_q);
        auto result_k = std::make_shared<ov::op::v0::Result>(reshape_k);
        auto result_v = std::make_shared<ov::op::v0::Result>(reshape_v);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result_q, result_k, result_v}, ov::ParameterVector{input});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
}

TEST_F(TransformationTestsF, FullyConnectedHorizontalFusion_gguf_mismatched_quant_types_no_fusion) {
    // V uses a different GGUF quant type than Q/K -- the group must not fuse.
    const size_t k = 4096;
    std::vector<int64_t> pattern = {7, -1};
    auto weight_q = make_gguf_q4_1_weight(4096, k, 0x10);
    auto weight_k = make_gguf_q4_1_weight(1024, k, 0x40);
    const auto& t0 = ov::element::gguf_q4_0;
    const size_t blocks_per_row = k / t0.block_elem_count();
    std::vector<uint8_t> v_data(1024 * blocks_per_row * t0.block_byte_size(), 0x80);
    auto weight_v = std::make_shared<ov::op::v0::Constant>(t0, ov::Shape{1024, k}, v_data.data());
    auto build_model = [&]() {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 7, 4096});
        auto scale_q = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}, std::vector<float>{1.0f});
        auto scale_k = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}, std::vector<float>{1.0f});
        auto scale_v = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}, std::vector<float>{1.0f});
        auto bias_q = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias_k = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto bias_v = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto zp_q = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto zp_k = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto zp_v = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_q = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_q, bias_q, scale_q, zp_q);
        auto fc_k = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_k, bias_k, scale_k, zp_k);
        auto fc_v = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weight_v, bias_v, scale_v, zp_v);
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, pattern);
        auto reshape_q = std::make_shared<ov::op::v1::Reshape>(fc_q, reshape_pattern, true);
        auto reshape_k = std::make_shared<ov::op::v1::Reshape>(fc_k, reshape_pattern, true);
        auto reshape_v = std::make_shared<ov::op::v1::Reshape>(fc_v, reshape_pattern, true);
        auto result_q = std::make_shared<ov::op::v0::Result>(reshape_q);
        auto result_k = std::make_shared<ov::op::v0::Result>(reshape_k);
        auto result_v = std::make_shared<ov::op::v0::Result>(reshape_v);
        return std::make_shared<ov::Model>(ov::ResultVector{result_q, result_k, result_v}, ov::ParameterVector{input});
    };
    model = build_model();
    manager.register_pass<FullyConnectedHorizontalFusion>();
}
}  // namespace intel_gpu
}  // namespace test
}  // namespace ov