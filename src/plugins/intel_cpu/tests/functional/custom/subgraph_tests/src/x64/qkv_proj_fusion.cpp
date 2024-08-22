// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

struct QKVProjFusionParams {
    size_t batch;
    size_t len;
    size_t hidden;
    size_t q_proj_size;
    size_t k_proj_size;
    size_t v_proj_size;
};

class QKVProjFusionTest : public testing::WithParamInterface<QKVProjFusionParams>,
                          public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QKVProjFusionParams>& obj) {
        std::ostringstream result;
        result << "batch=" << obj.param.batch << "_";
        result << "len=" << obj.param.len << "_";
        result << "hidden=" << obj.param.hidden << "_";
        result << "q_proj_size=" << obj.param.q_proj_size << "_";
        result << "k_proj_size=" << obj.param.k_proj_size << "_";
        result << "v_proj_size=" << obj.param.v_proj_size << "_";
        result << obj.index;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        auto& param = this->GetParam();

        configuration[ov::hint::inference_precision.name()] = "bf16";

        std::vector<size_t> shape_in = {param.batch, param.len, param.hidden};

        auto src = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(shape_in));

        std::vector<float> q_proj_w(param.q_proj_size * param.hidden, 0);
        std::vector<float> k_proj_w(param.k_proj_size * param.hidden, 0);
        std::vector<float> v_proj_w(param.v_proj_size * param.hidden, 0);

        for (auto& v : q_proj_w) v = ((std::rand() & 255) - 128)/128.0f;
        for (auto& v : k_proj_w) v = ((std::rand() & 255) - 128)/128.0f;
        for (auto& v : v_proj_w) v = ((std::rand() & 255) - 128)/128.0f;

        auto q_proj_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{param.q_proj_size, param.hidden}, q_proj_w);
        auto k_proj_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{param.k_proj_size, param.hidden}, k_proj_w);
        auto v_proj_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{param.v_proj_size, param.hidden}, v_proj_w);

        auto q_proj = std::make_shared<ov::op::v0::MatMul>(src, q_proj_weight, false, true);
        auto k_proj = std::make_shared<ov::op::v0::MatMul>(src, k_proj_weight, false, true);
        auto v_proj = std::make_shared<ov::op::v0::MatMul>(src, v_proj_weight, false, true);

        function = std::make_shared<ov::Model>(ov::NodeVector{q_proj, k_proj, v_proj}, ov::ParameterVector{src});
    }
    void check_results() {
        auto exec_model = compiledModel.get_runtime_model();

        int fused_node_found = 0;
        for (const auto& n : exec_model->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (layer_type == "QKVProjection")
                fused_node_found++;
        }
        ASSERT_EQ(fused_node_found, 1);
    }
};

TEST_P(QKVProjFusionTest, CompareWithRefs) {
    if (!ov::with_cpu_x86_avx512_core_amx_bf16())
        GTEST_SKIP();
    run();
    check_results();
}

namespace {

const std::vector<QKVProjFusionParams> qkv_params = {
    {5, 37, 4096, 4096, 4096, 4096},       // Llama-7B
    {7, 37, 3584, 128*28, 128*4, 128*4},   // Qwen2-7B: hidden_size_per_head:128, num_attention_heads:28, num_key_value_heads:4
};

INSTANTIATE_TEST_SUITE_P(smoke_QKVProjFusion,
                         QKVProjFusionTest,
                         ::testing::ValuesIn(qkv_params),
                         QKVProjFusionTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
