// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include "common_test_utils/ov_test_utils.hpp"
#include <transformations/utils/utils.hpp>

#include <plugin/transformations/kv_cache_fusion.hpp>

#include "intel_gpu/op/read_value.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/variadic_split.hpp"
#include "intel_gpu/op/kv_cache.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, KVCacheFusionTest1) {
    {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto past = std::make_shared<ov::op::v6::ReadValue>(variable);
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 32, -1, 80});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past, parameter}, 2);
        auto present = std::make_shared<ov::op::v6::Assign>(concat, variable);
        auto result = std::make_shared<ov::op::v0::Result>(concat);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::SinkVector{present}, ov::ParameterVector{parameter});
        manager.register_pass<KVCacheFusion>();
    }
    {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 32, -1, 80});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(past, parameter, variable, 2, ov::element::f16);
        auto result = std::make_shared<ov::op::v0::Result>(kv_cache);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    }
}

TEST_F(TransformationTestsF, KVCacheFusionTest2) {
    {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f32, "v0"});
        auto past = std::make_shared<ov::op::v6::ReadValue>(variable);
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 80});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto gather_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, axis);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather_past, parameter}, 2);
        auto present = std::make_shared<ov::op::v6::Assign>(concat, variable);
        auto result = std::make_shared<ov::op::v0::Result>(concat);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::SinkVector{present}, ov::ParameterVector{parameter, beam_idx});
        manager.register_pass<KVCacheFusion>();
    }
    {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f32, "v0"});
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 80});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto gather_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, axis);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(gather_past, parameter, variable, 2, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(kv_cache);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter, beam_idx});
    }
}

// Capturing a case when 1st input to Concat is a Node with multiple outputs
TEST_F(TransformationTestsF, KVCacheFusionTest3) {
        {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f32, "v0"});
        auto past = std::make_shared<ov::op::v6::ReadValue>(variable);
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 80});
        auto var_split = std::make_shared<ov::op::v1::VariadicSplit>(parameter,
                                                                     ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1}),
                                                                     ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {32}));
                                                                     std::cout << var_split->get_output_partial_shape(0) << std::endl;
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto gather_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, axis);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather_past, var_split->output(0)}, 2);
        auto present = std::make_shared<ov::op::v6::Assign>(concat, variable);
        auto result = std::make_shared<ov::op::v0::Result>(concat);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::SinkVector{present}, ov::ParameterVector{parameter, beam_idx});
        manager.register_pass<KVCacheFusion>();
    }
    {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f32, "v0"});
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, -1, 80});
        auto var_split = std::make_shared<ov::op::v1::VariadicSplit>(parameter,
                                                                     ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1}),
                                                                     ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {32}));
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto gather_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, axis);
        auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(gather_past, var_split->output(0), variable, 2, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(kv_cache);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter, beam_idx});
    }
}
