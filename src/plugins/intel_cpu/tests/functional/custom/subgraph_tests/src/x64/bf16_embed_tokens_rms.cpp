// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <functional>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace ov::test {

// Here we test a common pattern in LLMs - Token embeddings selection.
// In this particular tests embeddings are also quantized to u8.
// The idea is that such a subgraph (before the attention) must be preserved in fp32 to avoid accuracy loss.

class BF16EmbedTokensRMS : public SubgraphBaseTest {
public:
    void SetUp() override {
        const std::vector<InputShape> input_shapes = {
            {{-1, -1}, {{1, 32}, {2, 32}}},  // param 0
            {{-1, -1, EMBEDDINGS_SIZE}, {{1, 32, EMBEDDINGS_SIZE}, {2, 32, EMBEDDINGS_SIZE}}}};

        init_input_shapes(input_shapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1])};

        auto decompression_subgraph =
            utils::initGatherDecompressionSubgraph(ov::Shape{DICTIONARY_SIZE, EMBEDDINGS_SIZE},
                                                   -1 /*disable group quantization*/,
                                                   ov::element::u8 /*data prc*/,
                                                   ov::element::f32 /*output prc*/,
                                                   true /*add_subtract*/,
                                                   false /*add reshape*/,
                                                   false /*per tensor zp*/,
                                                   false /*per tensor scale*/);

        auto gather_axis = utils::make_constant(ov::element::i32, {1}, std::vector<int>{0});

        auto gather = std::make_shared<ov::op::v8::Gather>(decompression_subgraph, params.front(), gather_axis);

        auto sum = std::make_shared<ov::op::v1::Add>(gather, params[1]);

        // RMS pattern
        auto sqr = utils::make_constant(ov::element::f32, ov::Shape{1, 1, 1}, std::vector<float>{2.f});
        sqr = std::make_shared<ov::op::v1::Power>(sum, sqr);
        auto reduce_mean = std::make_shared<ov::op::v1::ReduceMean>(
            sqr,
            utils::make_constant(ov::element::i32, ov::Shape{1}, std::vector<int>{-1}),
            true /*keep dims*/);

        auto add_op = std::make_shared<ov::op::v1::Add>(
            reduce_mean,
            utils::make_constant(ov::element::f32, ov::Shape{1, 1, 1}, std::vector<float>{1.f}));

        auto sqrt_op = std::make_shared<ov::op::v0::Sqrt>(add_op);
        auto one = utils::make_constant(ov::element::f32, ov::Shape{1, 1, 1}, std::vector<float>{1});

        auto divide = std::make_shared<ov::op::v1::Divide>(one, sqrt_op);

        auto norm = std::make_shared<ov::op::v1::Multiply>(sum, divide);

        auto gamma = utils::make_constant(ov::element::f32, ov::Shape{1, 1, EMBEDDINGS_SIZE});

        norm = std::make_shared<ov::op::v1::Multiply>(norm, gamma);

        // here we insert FC layer to trigger BF16 markup

        auto fc_weights = utils::make_constant(ov::element::f32, ov::Shape{EMBEDDINGS_SIZE, EMBEDDINGS_SIZE});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(norm, fc_weights);

        function = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(matmul)},
                                               params,
                                               "BF16_EmbedTokens_RMS");
        targetDevice = utils::DEVICE_CPU;

        rel_threshold = abs_threshold = 0.002;

        configuration.insert({ov::hint::inference_precision.name(), ov::element::bf16});
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];

            ov::Tensor tensor;

            if (funcInput.get_element_type().is_integral()) {
                utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = DICTIONARY_SIZE;
                in_data.resolution = 1;
                tensor =
                    utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            } else {
                utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = 10;
                in_data.resolution = 1000;
                tensor =
                    utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

private:
    void check_subgraph_precision() {
        auto runtime_model = compiledModel.get_runtime_model();
        for (const auto& node : runtime_model->get_ordered_ops()) {
            if (node->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>() == "Gather") {
                GTEST_ASSERT_EQ(node->get_output_element_type(0), ov::element::f32)
                    << "Gather node output type is expected to be fp32";

                std::function<void(const ov::Node*)> check_fp32_path;
                check_fp32_path = [&check_fp32_path](const ov::Node* node) {
                    const auto& outputs = node->get_output_target_inputs(0);
                    if (outputs.empty()) {
                        // we reached output
                        return;
                    }
                    auto* next_node = outputs.begin()->get_node();
                    if (node->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>() == "Reorder") {
                        return;
                    }
                    GTEST_ASSERT_EQ(node->get_output_element_type(0), ov::element::f32)
                        << node->get_type_name() << " node output type is expected to be fp32";
                    check_fp32_path(next_node);
                };
                check_fp32_path(node.get());
            }
        }
    }

private:
    static constexpr size_t DICTIONARY_SIZE = 512;
    static constexpr size_t EMBEDDINGS_SIZE = 128;
};

TEST_F(BF16EmbedTokensRMS, smoke_BF16EmbedTokensRMS_CPU) {
    if (!ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP();
    }
    run();
}

}  // namespace ov::test