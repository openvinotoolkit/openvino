// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "plugin/transformations/convert_moe_to_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {
TEST_F(TransformationTestsF, ConvertMOEToMOE3GemmCompressedTest) {
    disable_rt_info_check();
    {
        // tokens:32, hidden_size:2048, iter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{128, 1, 32, 1});
        auto routing_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{32, 8});

        // Gate projection
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 1}, {0});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 768, 16, 1}, {0.01f});
        auto reshape_const_gate = op::v0::Constant::create(element::i32, Shape{3}, {128, 768, 2048});

        auto w_gate_f16 = std::make_shared<op::v0::Convert>(wei_gate, element::f16);
        auto zp_gate_f16 = std::make_shared<op::v0::Convert>(zp_gate, element::f16);
        auto sub_gate = std::make_shared<op::v1::Subtract>(w_gate_f16, zp_gate_f16);
        auto mul_gate = std::make_shared<op::v1::Multiply>(sub_gate, scale_gate);
        auto reshape_gate = std::make_shared<op::v1::Reshape>(mul_gate, reshape_const_gate, false);
        auto convert_gate = std::make_shared<op::v0::Convert>(reshape_gate, element::f32);

        // Up projection
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 1}, {0});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 768, 16, 1}, {0.01f});
        auto reshape_const_up = op::v0::Constant::create(element::i32, Shape{3}, {128, 768, 2048});

        auto w_up_f16 = std::make_shared<op::v0::Convert>(wei_up, element::f16);
        auto zp_up_f16 = std::make_shared<op::v0::Convert>(zp_up, element::f16);
        auto sub_up = std::make_shared<op::v1::Subtract>(w_up_f16, zp_up_f16);
        auto mul_up = std::make_shared<op::v1::Multiply>(sub_up, scale_up);
        auto reshape_up = std::make_shared<op::v1::Reshape>(mul_up, reshape_const_up, false);
        auto convert_up = std::make_shared<op::v0::Convert>(reshape_up, element::f32);

        // Down projection
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 1}, {0});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 2048, 6, 1}, {0.01f});
        auto reshape_const_down = op::v0::Constant::create(element::i32, Shape{3}, {128, 2048, 768});

        auto wei_down_f16 = std::make_shared<op::v0::Convert>(wei_down, element::f16);
        auto zp_down_f16 = std::make_shared<op::v0::Convert>(zp_down, element::f16);
        auto sub_down = std::make_shared<op::v1::Subtract>(wei_down_f16, zp_down_f16);
        auto mul_down = std::make_shared<op::v1::Multiply>(sub_down, scale_down);
        auto reshape_down = std::make_shared<op::v1::Reshape>(mul_down, reshape_const_down, false);
        auto convert_down = std::make_shared<op::v0::Convert>(reshape_down, element::f32);

        // Construct MOE node
        ov::op::internal::MOE::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        auto moe = std::make_shared<ov::op::internal::MOE>(
            ov::OutputVector{hidden_states, routing_weights, routing_idx, convert_gate, convert_up, convert_down}, config);
        model = std::make_shared<ov::Model>(moe, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
        manager.register_pass<ConvertMOEToMOECompressed>(0);
    }
    {
        // Inputs
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{128, 1, 32, 1});
        auto routing_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{32, 8});

        // Gate and up projection
        auto reshape_const_gate_up = op::v0::Constant::create(element::i32, Shape{3}, {128, 768, 16});
        auto transpose_const_gate_up = op::v0::Constant::create(element::i32, Shape{3}, {0, 2, 1});

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 1}, {0});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 768, 16, 1}, {0.01f});
        auto zp_reshape_gate = std::make_shared<op::v1::Reshape>(zp_gate, reshape_const_gate_up, false);
        auto zp_transpose_gate = std::make_shared<ov::op::v1::Transpose>(zp_reshape_gate, transpose_const_gate_up);
        auto scale_reshape_gate = std::make_shared<op::v1::Reshape>(scale_gate, reshape_const_gate_up, false);
        auto scale_transpose_gate = std::make_shared<ov::op::v1::Transpose>(scale_reshape_gate, transpose_const_gate_up);

        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 1}, {0});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 768, 16, 1}, {0.01f});
        auto zp_reshape_up = std::make_shared<op::v1::Reshape>(zp_up, reshape_const_gate_up, false);
        auto zp_transpose_up = std::make_shared<ov::op::v1::Transpose>(zp_reshape_up, transpose_const_gate_up);
        auto scale_reshape_up = std::make_shared<op::v1::Reshape>(scale_up, reshape_const_gate_up, false);
        auto scale_transpose_up = std::make_shared<ov::op::v1::Transpose>(scale_reshape_up, transpose_const_gate_up);

        // Down projection
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 1}, {0});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 2048, 6, 1}, {0.01f});
        auto reshape_const_down = op::v0::Constant::create(element::i32, Shape{3}, {128, 2048, 6});
        auto transpose_const_down = op::v0::Constant::create(element::i32, Shape{3}, {0, 2, 1});
        auto zp_reshape_down = std::make_shared<op::v1::Reshape>(zp_down, reshape_const_down, false);
        auto zp_transpose_down = std::make_shared<ov::op::v1::Transpose>(zp_reshape_down, transpose_const_down);
        auto scale_reshape_down = std::make_shared<op::v1::Reshape>(scale_down, reshape_const_down, false);
        auto scale_transpose_down = std::make_shared<ov::op::v1::Transpose>(scale_reshape_down, transpose_const_down);

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.top_k = 8;
        config.group_size = 128;
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{hidden_states, routing_weights, routing_idx,
                wei_gate, scale_transpose_gate, zp_transpose_gate,
                wei_up, scale_transpose_up, zp_transpose_up,
                wei_down, scale_transpose_down, zp_transpose_down}, config);
        model_ref = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
    }
}
}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
