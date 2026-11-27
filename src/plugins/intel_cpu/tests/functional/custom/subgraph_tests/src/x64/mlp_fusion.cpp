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
    // Combined gate_up weight + VariadicSplit options (the combined pattern is also used when use_swapped_outputs)
    bool use_combined_gate_up = false;
    ov::element::Type gate_up_weight_type = ov::element::f16;
    ov::element::Type split_lengths_type = ov::element::i32;
    bool use_positive_split_axis = false;  // rank - 1 instead of -1
    bool use_unequal_split = false;        // {up_size, 1} halves: must not fuse
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
        if (obj.param.use_combined_gate_up || obj.param.use_swapped_outputs) {
            result << "gate_up_weight_type=" << obj.param.gate_up_weight_type << "_";
            result << "split_lengths_type=" << obj.param.split_lengths_type << "_";
            result << "use_positive_split_axis=" << obj.param.use_positive_split_axis << "_";
            result << "use_unequal_split=" << obj.param.use_unequal_split << "_";
        }
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

        if (param.use_combined_gate_up || param.use_swapped_outputs) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -0.5;
            in_data.range = 1.0;
            in_data.resolution = 16;

            // The unequal case splits {up_size, 1}: the size-1 half broadcasts in the Multiply, so the graph stays
            // valid, but LLMMLP assumes equal halves and must not fuse it.
            const size_t second_size = param.use_unequal_split ? 1 : param.up_size;

            // Combined gate_up weight in FP16 (or BF16) format
            auto tensor =
                ov::test::utils::create_and_fill_tensor(param.gate_up_weight_type,
                                                        ov::Shape{param.up_size + second_size, param.down_size},
                                                        in_data);
            auto gate_up_weight = std::make_shared<ov::op::v0::Constant>(tensor);
            auto gate_up_weight_f32 = std::make_shared<ov::op::v0::Convert>(gate_up_weight, ov::element::f32);
            // Mark as decompression to prevent constant folding optimization and avoid pattern mismatch
            mark_as_decompression(gate_up_weight_f32);

            auto gate_up_proj = std::make_shared<ov::op::v0::MatMul>(src, gate_up_weight_f32, false, true);

            auto split_lengths = ov::op::v0::Constant::create(param.split_lengths_type,
                                                              ov::Shape{2},
                                                              std::vector<size_t>{param.up_size, second_size});
            const int64_t axis = param.use_positive_split_axis ? inputDynamicShapes[0].rank().get_length() - 1 : -1;
            auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            auto gate_up_split = std::make_shared<ov::op::v1::VariadicSplit>(gate_up_proj, axis_const, split_lengths);

            // Swapped outputs test the COMBINED_UP_GATE type: activation on output[1], up branch from output[0]
            auto gate_part = gate_up_split->output(param.use_swapped_outputs ? 1 : 0);
            if (param.act_type == "Swish")
                gate_act = std::make_shared<ov::op::v4::Swish>(gate_part);
            if (param.act_type == "Gelu")
                gate_act = std::make_shared<ov::op::v7::Gelu>(gate_part);

            up_output = gate_up_split->output(param.use_swapped_outputs ? 0 : 1);
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

        if (GetParam().use_unequal_split) {
            ASSERT_EQ(fused_node_found, 0) << "Unequal gate/up halves must not fuse";
        } else {
            // Both normal and swapped cases should fuse successfully
            ASSERT_EQ(fused_node_found, 1)
                << "Fusion should occur with valid MLP patterns (both normal and swapped cases)";
        }
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
// Rank-2 [tokens, hidden] activations, as produced by vLLM
static ov::test::InputShape ishape_2d{ov::PartialShape{-1, 4096 / 4},
                                      {ov::Shape{8, 4096 / 4}, ov::Shape{185, 4096 / 4}}};

const std::vector<LLMMLPFusionParams> mlp_params = {
    // Standard separate weights cases (should all fuse successfully)
    {ishape, 4096 / 4, 11008 / 4, "Gelu", false, false},
    {ishape, 4096 / 4, 11008 / 4, "Gelu", true, false},
    {ishape, 4096 / 4, 11008 / 4, "Swish", false, false},
    {ishape, 4096 / 4, 11008 / 4, "Swish", true, false},

    // Test case with swapped VariadicSplit outputs (should fuse with COMBINED_UP_GATE type)
    {ishape, 4096 / 4, 11008 / 4, "Gelu", false, true},

    // Rank-2 input, separate and combined gate_up weights
    {ishape_2d, 4096 / 4, 11008 / 4, "Swish", false, false},
    {ishape_2d, 4096 / 4, 11008 / 4, "Swish", false, false, true},

    // Combined gate_up split with a positive axis and i64 lengths (the form common optimizations produce)
    {ishape, 4096 / 4, 11008 / 4, "Swish", false, false, true, ov::element::f16, ov::element::i64, true},
    {ishape_2d, 4096 / 4, 11008 / 4, "Swish", false, true, true, ov::element::f16, ov::element::i64, true},

    // Combined bf16 gate_up weights
    {ishape, 4096 / 4, 11008 / 4, "Swish", false, false, true, ov::element::bf16},
    {ishape_2d, 4096 / 4, 11008 / 4, "Swish", false, false, true, ov::element::bf16, ov::element::i64, true},

    // Negative: unequal split halves must not fuse
    {ishape, 4096 / 4, 11008 / 4, "Swish", false, false, true, ov::element::f16, ov::element::i32, false, true},
    {ishape_2d, 4096 / 4, 11008 / 4, "Swish", false, false, true, ov::element::bf16, ov::element::i64, true, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_LLMMLPFusion,
                         LLMMLPFusionTest,
                         ::testing::ValuesIn(mlp_params),
                         LLMMLPFusionTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
