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
    bool use_dynamic_quant;
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
        result << "use_dynamic_quant=" << obj.param.use_dynamic_quant << "_";
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

        auto create_const = [&](size_t OC, size_t IC) -> std::shared_ptr<ov::Node> {
            if (param.use_dynamic_quant) {
                ov::test::utils::InputGenerateData in_data;
                // range [-128, +127]
                in_data.start_from = -128;
                in_data.range = 256;
                in_data.resolution = 256;
                auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::i8, ov::Shape{OC, IC}, in_data);
                auto weight_const_i8 = std::make_shared<ov::op::v0::Constant>(tensor);
                auto weight_const_f32 = std::make_shared<ov::op::v0::Convert>(weight_const_i8, ov::element::f32);

                // range after dequantize, [-1, +1]
                in_data.start_from = 0;
                in_data.range = 1;
                in_data.resolution = 128;
                auto tensor_scale_per_oc = ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{OC, 1}, in_data);
                auto scale_per_oc = std::make_shared<ov::op::v0::Constant>(tensor_scale_per_oc);

                auto weight_deq = std::make_shared<ov::op::v1::Multiply>(weight_const_f32, scale_per_oc);
                return weight_deq;
            }
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -0.5;
            in_data.range = 1;
            in_data.resolution = 128;
            auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{OC, IC}, in_data);
            return std::make_shared<ov::op::v0::Constant>(tensor);
        };
        if (param.use_dynamic_quant)
            configuration.insert({ov::hint::dynamic_quantization_group_size.name(), std::numeric_limits<uint64_t>::max()});

        auto q_proj_weight = create_const(param.q_proj_size, param.hidden);
        auto k_proj_weight = create_const(param.k_proj_size, param.hidden);
        auto v_proj_weight = create_const(param.v_proj_size, param.hidden);

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
static ov::test::InputShape ishape_llama2_7b{ov::PartialShape{-1, -1, 4096 / 4}, {ov::Shape{1, 8, 4096 / 4}, ov::Shape{5, 7, 4096 / 4}}};
static ov::test::InputShape ishape_qwen2_7b{ov::test::InputShape{ov::PartialShape{-1, -1, 3584 / 2}, {ov::Shape{1, 8, 3584 / 2}, ov::Shape{5, 7, 3584 / 2}}}};

const std::vector<QKVProjFusionParams> qkv_params = {
    // Llama-7B with reduced size
    {ishape_llama2_7b,  4096 / 4, 4096 / 4, 4096 / 4, 4096 / 4, false},
    {ishape_llama2_7b,  4096 / 4, 4096 / 4, 4096 / 4, 4096 / 4, true},
    // Qwen2-7B: hidden_size_per_head:128, num_attention_heads:28, num_key_value_heads:4
    {ishape_qwen2_7b, 3584 / 2, 128 * 28 / 2, 128 * 4 / 2, 128 * 4 / 2, false},
    {ishape_qwen2_7b, 3584 / 2, 128 * 28 / 2, 128 * 4 / 2, 128 * 4 / 2, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_QKVProjFusion,
                         QKVProjFusionTest,
                         ::testing::ValuesIn(qkv_params),
                         QKVProjFusionTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
