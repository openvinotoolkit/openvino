// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

struct LLMMLPFusionParams {
    ov::test::InputShape inputShape;
    size_t down_size;
    size_t up_size;
    std::string act_type;
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

        auto create_const = [](ov::Shape shape, int resolution) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -0.5;
            in_data.range = 1;
            in_data.resolution = resolution;
            auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, in_data);
            return std::make_shared<ov::op::v0::Constant>(tensor);
        };

        auto gate_weight = create_const(ov::Shape{param.up_size, param.down_size}, 100);
        auto up_weight = create_const(ov::Shape{param.up_size, param.down_size}, 100);
        // down_proj has special cache blocking along K dimension requires lower weight resolution
        auto down_weight = create_const(ov::Shape{param.down_size, param.up_size}, 16);

        auto gate_proj = std::make_shared<ov::op::v0::MatMul>(src, gate_weight, false, true);
        auto up_proj = std::make_shared<ov::op::v0::MatMul>(src, up_weight, false, true);

        std::shared_ptr<Node> gate_act;
        if (param.act_type == "Swish")
            gate_act = std::make_shared<ov::op::v4::Swish>(gate_proj);
        if (param.act_type == "Gelu")
            gate_act = std::make_shared<ov::op::v7::Gelu>(gate_proj);

        auto gate_up = std::make_shared<ov::op::v1::Multiply>(gate_act, up_proj);
        auto output = std::make_shared<ov::op::v0::MatMul>(gate_up, down_weight, false, true);

        function = std::make_shared<ov::Model>(ov::NodeVector{output}, ov::ParameterVector{src});
    }

    void check_results() {
        auto exec_model = compiledModel.get_runtime_model();

        int fused_node_found = 0;
        for (const auto& n : exec_model->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (layer_type == "LLMMLP")
                fused_node_found++;
        }
        ASSERT_EQ(fused_node_found, 1);
    }
};

TEST_P(LLMMLPFusionTest, CompareWithRefs) {
    if (!ov::with_cpu_x86_avx512_core_amx_bf16())
        GTEST_SKIP();
    run();
    check_results();
}

namespace {

const std::vector<LLMMLPFusionParams> mlp_params = {
    {ov::test::InputShape{ov::PartialShape{-1, -1, 4096 / 4}, {ov::Shape{1, 8, 4096 / 4}, ov::Shape{5, 37, 4096 / 4}}},
     4096 / 4,
     11008 / 4,
     "Gelu"},
    {ov::test::InputShape{ov::PartialShape{-1, -1, 4096 / 4}, {ov::Shape{1, 8, 4096 / 4}, ov::Shape{5, 37, 4096 / 4}}},
     4096 / 4,
     11008 / 4,
     "Swish"},
};

INSTANTIATE_TEST_SUITE_P(smoke_LLMMLPFusion,
                         LLMMLPFusionTest,
                         ::testing::ValuesIn(mlp_params),
                         LLMMLPFusionTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
