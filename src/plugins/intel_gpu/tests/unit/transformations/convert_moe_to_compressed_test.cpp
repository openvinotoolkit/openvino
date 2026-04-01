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
#include "openvino/op/swish.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"

#include "openvino/op/sigmoid.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {
class ConvertMOEToMOE3GemmCompressedTest : public TransformationTestsF, public WithParamInterface<ov::element::Type> {
public:
    static std::string get_test_case_name(testing::TestParamInfo<ov::element::Type> obj) {
        ov::element::Type element_type = obj.param;
        std::ostringstream result;
        result << "ElementType=" << element_type.get_type_name();
        return result.str();
    }
};

TEST_P(ConvertMOEToMOE3GemmCompressedTest, GEMM3_SwiGLU) {
    disable_rt_info_check();
    const auto& data_type = GetParam();
    {
        // tokens:32, hidden_size:2048, iter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(data_type, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(data_type, Shape{128, 1, 32, 1});
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
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(data_type, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(data_type, Shape{128, 1, 32, 1});
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
        std::shared_ptr<ov::Node> moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{hidden_states, routing_weights, routing_idx,
                wei_gate, scale_transpose_gate, zp_transpose_gate,
                wei_up, scale_transpose_up, zp_transpose_up,
                wei_down, scale_transpose_down, zp_transpose_down}, config);
        if (config.out_type != data_type) {
            moe_compressed = std::make_shared<ov::op::v0::Convert>(moe_compressed, data_type);
        }
        model_ref = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ConvertMOEToMOE3GemmCompressedTest,
                         Values(element::f16, element::f32),
                         ConvertMOEToMOE3GemmCompressedTest::get_test_case_name);

TEST_F(TransformationTestsF, ConvertMOEToMOE3GemmSharedExpertCompressedTest) {
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

        // Shared expert Gate projection (separate weights, 3D without num_experts dim)
        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{768, 16, 128}, {2});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{768, 16, 1}, {1});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{768, 16, 1}, {0.02f});
        auto sh_reshape_const_gate = op::v0::Constant::create(element::i32, Shape{2}, {768, 2048});
        auto sh_w_gate_f16 = std::make_shared<op::v0::Convert>(sh_wei_gate, element::f16);
        auto sh_zp_gate_f16 = std::make_shared<op::v0::Convert>(sh_zp_gate, element::f16);
        auto sh_sub_gate = std::make_shared<op::v1::Subtract>(sh_w_gate_f16, sh_zp_gate_f16);
        auto sh_mul_gate = std::make_shared<op::v1::Multiply>(sh_sub_gate, sh_scale_gate);
        auto sh_reshape_gate = std::make_shared<op::v1::Reshape>(sh_mul_gate, sh_reshape_const_gate, false);
        auto sh_convert_gate = std::make_shared<op::v0::Convert>(sh_reshape_gate, element::f32);

        // Shared expert Up projection (separate weights, 3D without num_experts dim)
        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{768, 16, 128}, {2});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{768, 16, 1}, {1});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{768, 16, 1}, {0.02f});
        auto sh_reshape_const_up = op::v0::Constant::create(element::i32, Shape{2}, {768, 2048});
        auto sh_w_up_f16 = std::make_shared<op::v0::Convert>(sh_wei_up, element::f16);
        auto sh_zp_up_f16 = std::make_shared<op::v0::Convert>(sh_zp_up, element::f16);
        auto sh_sub_up = std::make_shared<op::v1::Subtract>(sh_w_up_f16, sh_zp_up_f16);
        auto sh_mul_up = std::make_shared<op::v1::Multiply>(sh_sub_up, sh_scale_up);
        auto sh_reshape_up = std::make_shared<op::v1::Reshape>(sh_mul_up, sh_reshape_const_up, false);
        auto sh_convert_up = std::make_shared<op::v0::Convert>(sh_reshape_up, element::f32);

        // Shared expert Down projection (separate weights, 3D without num_experts dim)
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{2048, 6, 128}, {2});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{2048, 6, 1}, {1});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{2048, 6, 1}, {0.02f});
        auto sh_reshape_const_down = op::v0::Constant::create(element::i32, Shape{2}, {2048, 768});
        auto sh_w_down_f16 = std::make_shared<op::v0::Convert>(sh_wei_down, element::f16);
        auto sh_zp_down_f16 = std::make_shared<op::v0::Convert>(sh_zp_down, element::f16);
        auto sh_sub_down = std::make_shared<op::v1::Subtract>(sh_w_down_f16, sh_zp_down_f16);
        auto sh_mul_down = std::make_shared<op::v1::Multiply>(sh_sub_down, sh_scale_down);
        auto sh_reshape_down = std::make_shared<op::v1::Reshape>(sh_mul_down, sh_reshape_const_down, false);
        auto sh_convert_down = std::make_shared<op::v0::Convert>(sh_reshape_down, element::f32);

        // Construct MOE node
        ov::op::internal::MOE::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        auto hidden_states_f32 = std::make_shared<ov::op::v0::Convert>(hidden_states, element::f32);
        auto moe = std::make_shared<ov::op::internal::MOE>(
            ov::OutputVector{hidden_states_f32, routing_weights, routing_idx, convert_gate, convert_up, convert_down}, config);

        // In the actual model (Qwen3-Next), shared expert MatMuls use hidden_states
        // from a different node (after a Reshape) than MOE's hidden_states input.
        // matmul_experts_fusion extracts input_value(0) of the Reshape as MOE's hidden_states,
        // while the shared expert MatMuls take the Reshape output directly.
        auto reshape_const_hs = op::v0::Constant::create(element::i64, Shape{2}, {32, 2048});
        auto hidden_states_reshaped = std::make_shared<ov::op::v1::Reshape>(hidden_states_f32, reshape_const_hs, false);

        // Shared expert computation (using separate weights and reshaped hidden_states)
        auto shared_gate_m = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, sh_convert_gate, false, true);
        auto shared_swish_m = std::make_shared<ov::op::v4::Swish>(shared_gate_m);
        auto shared_up_m = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, sh_convert_up, false, true);
        auto shared_mul_m = std::make_shared<ov::op::v1::Multiply>(shared_swish_m, shared_up_m);
        auto shared_down_m = std::make_shared<ov::op::v0::MatMul>(shared_mul_m, sh_convert_down, false, true);
        auto shared_gate_gate_wei_m = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{2048, 1}, std::vector<float>(2048, 1.0f));
        auto shared_gate_gate_m = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, shared_gate_gate_wei_m);
        auto shared_gate_sigmoid_m = std::make_shared<ov::op::v0::Sigmoid>(shared_gate_gate_m);
        auto shared_expert_gated_m = std::make_shared<ov::op::v1::Multiply>(shared_gate_sigmoid_m, shared_down_m);

        // In the actual model, there's a Reshape between the gated shared expert output and the Add
        auto reshape_const_output = op::v0::Constant::create(element::i64, Shape{2}, {32, 2048});
        auto shared_expert_reshaped = std::make_shared<ov::op::v1::Reshape>(shared_expert_gated_m, reshape_const_output, false);

        auto add_m = std::make_shared<ov::op::v1::Add>(shared_expert_reshaped, moe);

        model = std::make_shared<ov::Model>(add_m, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
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

        // Shared expert weights (3D without num_experts dim, separate from MOE expert weights)
        auto sh_reshape_const_gate_up = op::v0::Constant::create(element::i32, Shape{2}, {768, 16});
        auto sh_transpose_const_gate_up = op::v0::Constant::create(element::i32, Shape{2}, {1, 0});

        auto sh_wei_gate = op::v0::Constant::create(element::u4, Shape{768, 16, 128}, {2});
        auto sh_zp_gate = op::v0::Constant::create(element::u4, Shape{768, 16, 1}, {1});
        auto sh_scale_gate = op::v0::Constant::create(element::f16, Shape{768, 16, 1}, {0.02f});
        auto sh_zp_reshape_gate = std::make_shared<op::v1::Reshape>(sh_zp_gate, sh_reshape_const_gate_up, false);
        auto sh_zp_transpose_gate = std::make_shared<ov::op::v1::Transpose>(sh_zp_reshape_gate, sh_transpose_const_gate_up);
        auto sh_scale_reshape_gate = std::make_shared<op::v1::Reshape>(sh_scale_gate, sh_reshape_const_gate_up, false);
        auto sh_scale_transpose_gate = std::make_shared<ov::op::v1::Transpose>(sh_scale_reshape_gate, sh_transpose_const_gate_up);

        auto sh_wei_up = op::v0::Constant::create(element::u4, Shape{768, 16, 128}, {2});
        auto sh_zp_up = op::v0::Constant::create(element::u4, Shape{768, 16, 1}, {1});
        auto sh_scale_up = op::v0::Constant::create(element::f16, Shape{768, 16, 1}, {0.02f});
        auto sh_zp_reshape_up = std::make_shared<op::v1::Reshape>(sh_zp_up, sh_reshape_const_gate_up, false);
        auto sh_zp_transpose_up = std::make_shared<ov::op::v1::Transpose>(sh_zp_reshape_up, sh_transpose_const_gate_up);
        auto sh_scale_reshape_up = std::make_shared<op::v1::Reshape>(sh_scale_up, sh_reshape_const_gate_up, false);
        auto sh_scale_transpose_up = std::make_shared<ov::op::v1::Transpose>(sh_scale_reshape_up, sh_transpose_const_gate_up);

        auto sh_reshape_const_down = op::v0::Constant::create(element::i32, Shape{2}, {2048, 6});
        auto sh_transpose_const_down = op::v0::Constant::create(element::i32, Shape{2}, {1, 0});
        auto sh_wei_down = op::v0::Constant::create(element::u4, Shape{2048, 6, 128}, {2});
        auto sh_zp_down = op::v0::Constant::create(element::u4, Shape{2048, 6, 1}, {1});
        auto sh_scale_down = op::v0::Constant::create(element::f16, Shape{2048, 6, 1}, {0.02f});
        auto sh_zp_reshape_down = std::make_shared<op::v1::Reshape>(sh_zp_down, sh_reshape_const_down, false);
        auto sh_zp_transpose_down = std::make_shared<ov::op::v1::Transpose>(sh_zp_reshape_down, sh_transpose_const_down);
        auto sh_scale_reshape_down = std::make_shared<op::v1::Reshape>(sh_scale_down, sh_reshape_const_down, false);
        auto sh_scale_transpose_down = std::make_shared<ov::op::v1::Transpose>(sh_scale_reshape_down, sh_transpose_const_down);

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.num_shared_expert = 1;
        config.top_k = 8;
        config.group_size = 128;
        config.out_type = ov::element::f16;
        auto hidden_states_f32 = std::make_shared<ov::op::v0::Convert>(hidden_states, element::f32);
        auto shared_gate_gate_wei_m = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{2048, 1}, std::vector<float>(2048, 1.0f));
        std::shared_ptr<ov::Node> moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{hidden_states_f32, routing_weights, routing_idx,
                wei_gate, scale_transpose_gate, zp_transpose_gate,
                wei_up, scale_transpose_up, zp_transpose_up,
                wei_down, scale_transpose_down, zp_transpose_down,
                sh_wei_gate, sh_scale_transpose_gate, sh_zp_transpose_gate,
                sh_wei_up, sh_scale_transpose_up, sh_zp_transpose_up,
                sh_wei_down, sh_scale_transpose_down, sh_zp_transpose_down,
                shared_gate_gate_wei_m}, config);
        // MOECompressed outputs f16, but the input model outputs f32 (Add of f32 MOE + f32 shared expert),
        // so the transformation inserts a Convert(f16->f32) to preserve the original output type.
        moe_compressed = std::make_shared<ov::op::v0::Convert>(moe_compressed, element::f32);
        model_ref = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
