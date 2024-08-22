// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

struct LLMMLPFusionParams {
    size_t batch;
    size_t len;
    size_t down_size;
    size_t up_size;
    std::string act_type;
};

class LLMMLPFusionTest : public testing::WithParamInterface<LLMMLPFusionParams>,
                          public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LLMMLPFusionParams>& obj) {
        std::ostringstream result;
        result << "batch=" << obj.param.batch << "_";
        result << "len=" << obj.param.len << "_";
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

        std::vector<size_t> shape_in = {param.batch, param.len, param.down_size};

        auto src = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(shape_in));

        std::vector<float> gate_proj_w(param.up_size * param.down_size, 0);
        std::vector<float> up_proj_w(param.up_size * param.down_size, 0);
        std::vector<float> down_proj_w(param.up_size * param.down_size, 0);

        for (auto& v : gate_proj_w) v = ((std::rand() & 15) - 8)/8.0f;
        for (auto& v : up_proj_w) v = ((std::rand() & 15) - 8)/8.0f;
        for (auto& v : down_proj_w) v = ((std::rand() & 15) - 8)/8.0f;

        auto gate_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{param.up_size, param.down_size}, gate_proj_w);
        auto up_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{param.up_size, param.down_size}, up_proj_w);
        auto down_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{param.down_size, param.up_size}, down_proj_w);

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
    {5, 37, 4096, 11008, "Gelu"},
    {5, 37, 4096, 11008, "Swish"},
};

INSTANTIATE_TEST_SUITE_P(smoke_LLMMLPFusion,
                         LLMMLPFusionTest,
                         ::testing::ValuesIn(mlp_params),
                         LLMMLPFusionTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
