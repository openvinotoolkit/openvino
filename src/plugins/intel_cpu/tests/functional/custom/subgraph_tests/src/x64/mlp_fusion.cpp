// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace ov {
namespace test {

struct LLMMLPFusionParams {
    ov::test::InputShape inputShape;
    size_t down_size;
    size_t up_size;
    std::string act_type;
    bool use_dynamic_quant;
    bool use_swapped_outputs;  // true = create pattern with swapped VariadicSplit outputs (should still fuse)
};

class LLMMLPFusionTest : public testing::WithParamInterface<LLMMLPFusionParams>, public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LLMMLPFusionParams>& obj) {
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({obj.param.inputShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : obj.param.inputShape.second) {
            result << ov::test::utils::vec2str(shape);
            result << "_";
        }
        result << "down_size=" << obj.param.down_size << "_";
        result << "up_size=" << obj.param.up_size << "_";
        result << "act_type=" << obj.param.act_type << "_";
        result << "use_dynamic_quant=" << obj.param.use_dynamic_quant << "_";
        result << "use_swapped_outputs=" << obj.param.use_swapped_outputs << "_";
        result << obj.index;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        auto& param = this->GetParam();

        configuration[ov::hint::inference_precision.name()] = "bf16";

        init_input_shapes({param.inputShape});

        auto src = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);

        auto create_const = [&](size_t OC, size_t IC, int resolution) -> std::shared_ptr<ov::Node> {
            if (param.use_dynamic_quant) {
                ov::test::utils::InputGenerateData in_data;
                // range [-128, +127]
                in_data.start_from = -64;
                in_data.range = 63;
                in_data.resolution = 128;
                auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::i8, ov::Shape{OC, IC}, in_data);
                auto weight_const_i8 = std::make_shared<ov::op::v0::Constant>(tensor);
                auto weight_const_f32 = std::make_shared<ov::op::v0::Convert>(weight_const_i8, ov::element::f32);

                // range after dequantize, [-1, +1]
                in_data.start_from = 0;
                in_data.range = 1;
                in_data.resolution = 128;
                auto tensor_scale_per_oc =
                    ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{OC, 1}, in_data);
                auto scale_per_oc = std::make_shared<ov::op::v0::Constant>(tensor_scale_per_oc);

                auto weight_deq = std::make_shared<ov::op::v1::Multiply>(weight_const_f32, scale_per_oc);
                return weight_deq;
            }

            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -0.5;
            in_data.range = 1;
            in_data.resolution = resolution;
            auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{OC, IC}, in_data);
            return std::make_shared<ov::op::v0::Constant>(tensor);
        };
        if (param.use_dynamic_quant)
            configuration.insert(
                {ov::hint::dynamic_quantization_group_size.name(), std::numeric_limits<uint64_t>::max()});

        std::shared_ptr<Node> gate_act;
        ov::Output<ov::Node> up_output;

        if (param.use_swapped_outputs) {
            // Create pattern with swapped VariadicSplit outputs to test COMBINED_UP_GATE type
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -0.5;
            in_data.range = 1.0;
            in_data.resolution = 16;

            // Combined gate_up weight in FP16 format
            auto tensor_f16 = ov::test::utils::create_and_fill_tensor(ov::element::f16,
                                                                      ov::Shape{param.up_size * 2, param.down_size},
                                                                      in_data);
            auto gate_up_weight_f16 = std::make_shared<ov::op::v0::Constant>(tensor_f16);
            auto gate_up_weight_f32 = std::make_shared<ov::op::v0::Convert>(gate_up_weight_f16, ov::element::f32);
            // Mark as decompression to prevent constant folding optimization and avoid pattern mismatch
            mark_as_decompression(gate_up_weight_f32);

            auto gate_up_proj = std::make_shared<ov::op::v0::MatMul>(src, gate_up_weight_f32, false, true);

            auto split_lengths = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32,
                ov::Shape{2},
                std::vector<int32_t>{static_cast<int32_t>(param.up_size), static_cast<int32_t>(param.up_size)});
            auto axis_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, -1);
            auto gate_up_split = std::make_shared<ov::op::v1::VariadicSplit>(gate_up_proj, axis_const, split_lengths);

            // Swap outputs to test COMBINED_UP_GATE type
            auto gate_part = gate_up_split->output(1);  // activation on output[1]
            if (param.act_type == "Swish")
                gate_act = std::make_shared<ov::op::v4::Swish>(gate_part);
            if (param.act_type == "Gelu")
                gate_act = std::make_shared<ov::op::v7::Gelu>(gate_part);

            auto up_part = gate_up_split->output(0);  // up branch from output[0] (swapped case)
            up_output = up_part;
        } else {
            // Standard separate weights pattern
            auto gate_weight = create_const(param.up_size, param.down_size, 100);
            auto up_weight = create_const(param.up_size, param.down_size, 100);

            auto gate_proj = std::make_shared<ov::op::v0::MatMul>(src, gate_weight, false, true);
            auto up_proj = std::make_shared<ov::op::v0::MatMul>(src, up_weight, false, true);

            if (param.act_type == "Swish")
                gate_act = std::make_shared<ov::op::v4::Swish>(gate_proj);
            if (param.act_type == "Gelu")
                gate_act = std::make_shared<ov::op::v7::Gelu>(gate_proj);

            up_output = up_proj;
        }

        // Create compressed down projection weight
        ov::test::utils::InputGenerateData down_data;
        down_data.start_from = -0.5;
        down_data.range = 1;
        down_data.resolution = 16;
        auto tensor_f16_down = ov::test::utils::create_and_fill_tensor(ov::element::f16,
                                                                       ov::Shape{param.down_size, param.up_size},
                                                                       down_data);
        auto down_weight_f16 = std::make_shared<ov::op::v0::Constant>(tensor_f16_down);
        auto down_weight = std::make_shared<ov::op::v0::Convert>(down_weight_f16, ov::element::f32);

        auto gate_up = std::make_shared<ov::op::v1::Multiply>(gate_act, up_output);
        auto output = std::make_shared<ov::op::v0::MatMul>(gate_up, down_weight, false, true);

        function = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{src});
    }

    void check_results() {
        auto exec_model = compiledModel.get_runtime_model();
        int fused_node_found = 0;
        for (const auto& n : exec_model->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (layer_type == "LLMMLP")
                fused_node_found++;
        }

        // Both normal and swapped cases should fuse successfully
        ASSERT_EQ(fused_node_found, 1) << "Fusion should occur with valid MLP patterns (both normal and swapped cases)";
    }
};

TEST_P(LLMMLPFusionTest, CompareWithRefs) {
    if (!ov::with_cpu_x86_avx512_core_amx_bf16())
        GTEST_SKIP();
    run();
    check_results();
}

namespace {

static ov::test::InputShape ishape{ov::PartialShape{-1, -1, 4096 / 4},
                                   {ov::Shape{1, 8, 4096 / 4}, ov::Shape{5, 37, 4096 / 4}}};

const std::vector<LLMMLPFusionParams> mlp_params = {
    // Standard separate weights cases (should all fuse successfully)
    {ishape, 4096 / 4, 11008 / 4, "Gelu", false, false},
    {ishape, 4096 / 4, 11008 / 4, "Gelu", true, false},
    {ishape, 4096 / 4, 11008 / 4, "Swish", false, false},
    {ishape, 4096 / 4, 11008 / 4, "Swish", true, false},

    // Test case with swapped VariadicSplit outputs (should fuse with COMBINED_UP_GATE type)
    {ishape, 4096 / 4, 11008 / 4, "Gelu", false, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_LLMMLPFusion,
                         LLMMLPFusionTest,
                         ::testing::ValuesIn(mlp_params),
                         LLMMLPFusionTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
