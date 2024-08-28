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

struct QKVProjFusionParams {
    ov::test::InputShape inputShape;
    size_t hidden;
    size_t q_proj_size;
    size_t k_proj_size;
    size_t v_proj_size;
};

class QKVProjFusionTest : public testing::WithParamInterface<QKVProjFusionParams>,
                          public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QKVProjFusionParams>& obj) {
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({obj.param.inputShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : obj.param.inputShape.second) {
            result << ov::test::utils::vec2str(shape);
            result << "_";
        }
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

        auto q_proj_weight = create_const(ov::Shape{param.q_proj_size, param.hidden}, 128);
        auto k_proj_weight = create_const(ov::Shape{param.k_proj_size, param.hidden}, 128);
        auto v_proj_weight = create_const(ov::Shape{param.v_proj_size, param.hidden}, 128);

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

// the shape size is divided by a const to reduce test time
const std::vector<QKVProjFusionParams> qkv_params = {
    // Llama-7B
    {ov::test::InputShape{ov::PartialShape{-1, -1, 4096 / 4}, {ov::Shape{1, 8, 4096 / 4}, ov::Shape{5, 7, 4096 / 4}}},
     4096 / 4,
     4096 / 4,
     4096 / 4,
     4096 / 4},
    // Qwen2-7B: hidden_size_per_head:128, num_attention_heads:28, num_key_value_heads:4
    {ov::test::InputShape{ov::PartialShape{-1, -1, 3584 / 2}, {ov::Shape{1, 8, 3584 / 2}, ov::Shape{5, 7, 3584 / 2}}},
     3584 / 2,
     128 * 28 / 2,
     128 * 4 / 2,
     128 * 4 / 2},
};

INSTANTIATE_TEST_SUITE_P(smoke_QKVProjFusion,
                         QKVProjFusionTest,
                         ::testing::ValuesIn(qkv_params),
                         QKVProjFusionTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
