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

struct ActSparseFCFusionParams {
    ov::test::InputShape inputShape;
    size_t IC;
    size_t OC;
    ov::element::Type wtype;  // weight-type (f16, u8-asym, i8-sym, u4-asym, i4-sym)
    size_t IC_group_size;     // valid on INT4 case-only
    float sparsity_threshold;
};

class ActSparseFCFusionTest : public testing::WithParamInterface<ActSparseFCFusionParams>,
                              public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ActSparseFCFusionParams>& obj) {
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({obj.param.inputShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : obj.param.inputShape.second) {
            result << ov::test::utils::vec2str(shape);
            result << "_";
        }
        result << "wtype=" << obj.param.wtype << "_";
        result << "sparsity_threshold=" << obj.param.sparsity_threshold << "_";
        result << "OC=" << obj.param.OC << "_";
        result << "IC=" << obj.param.IC << "_";
        result << "IC_group_size=" << obj.param.IC_group_size << "_";
        result << obj.index;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        auto& param = this->GetParam();

        configuration[ov::hint::inference_precision.name()] = "f32";

        init_input_shapes({param.inputShape});

        auto src = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);

        auto create_const =
            [&](size_t OC, size_t IC, size_t IC_group_size, int resolution) -> std::shared_ptr<ov::Node> {
            if (param.wtype == ov::element::i8 || param.wtype == ov::element::i4) {
                ov::test::utils::InputGenerateData in_data;
                // range [-128, +127]
                in_data.start_from = -64;
                in_data.range = 63;
                in_data.resolution = 128;
                auto tensor = ov::test::utils::create_and_fill_tensor(
                    param.wtype,
                    (param.wtype == ov::element::i8) ? ov::Shape{OC, IC}
                                                     : ov::Shape{OC, IC / IC_group_size, IC_group_size},
                    in_data);
                auto weight_const_i8i4 = std::make_shared<ov::op::v0::Constant>(tensor);
                auto weight_const_f32 = std::make_shared<ov::op::v0::Convert>(weight_const_i8i4, ov::element::f32);

                // range after dequantize, [-1, +1]
                in_data.start_from = 0;
                in_data.range = 1;
                in_data.resolution = 128;
                auto tensor_scales = ov::test::utils::create_and_fill_tensor(
                    ov::element::f32,
                    (param.wtype == ov::element::i8) ? ov::Shape{OC, 1} : ov::Shape{OC, IC / IC_group_size, 1},

                    in_data);
                auto scales = std::make_shared<ov::op::v0::Constant>(tensor_scales);

                auto weight_deq = std::make_shared<ov::op::v1::Multiply>(weight_const_f32, scales);

                if (param.wtype == ov::element::i8)
                    return weight_deq;

                auto weight_shape = std::make_shared<ov::op::v0::Constant>(
                    ov::element::i32,
                    ov::Shape{2},
                    std::vector<int>{static_cast<int>(OC), static_cast<int>(IC)});
                return std::make_shared<ov::opset10::Reshape>(weight_deq, weight_shape, false);
            }
            if (param.wtype == ov::element::u8 || param.wtype == ov::element::u4) {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = (param.wtype == ov::element::u8) ? 255 : 15;
                in_data.resolution = 128;
                auto tensor = ov::test::utils::create_and_fill_tensor(
                    param.wtype,
                    (param.wtype == ov::element::u8) ? ov::Shape{OC, IC}
                                                     : ov::Shape{OC, IC / IC_group_size, IC_group_size},
                    in_data);
                auto weight_const_u8u4 = std::make_shared<ov::op::v0::Constant>(tensor);
                auto weight_const_f32 = std::make_shared<ov::op::v0::Convert>(weight_const_u8u4, ov::element::f32);

                in_data.start_from = (param.wtype == ov::element::u8) ? 128 : 8;
                in_data.range = 2;
                in_data.resolution = 128;
                auto tensor_zp = ov::test::utils::create_and_fill_tensor(
                    param.wtype,
                    (param.wtype == ov::element::u8) ? ov::Shape{OC, 1} : ov::Shape{OC, IC / IC_group_size, 1},
                    in_data);
                auto zero_point_const_u8u4 = std::make_shared<ov::op::v0::Constant>(tensor_zp);
                auto zero_point_const_f32 =
                    std::make_shared<ov::op::v0::Convert>(zero_point_const_u8u4, ov::element::f32);
                auto weight_f32 = std::make_shared<ov::opset10::Subtract>(weight_const_f32, zero_point_const_f32);

                // range after dequantize, [-1, +1]
                in_data.start_from = 0;
                in_data.range = 1;
                in_data.resolution = 128;
                auto tensor_scale_per_oc = ov::test::utils::create_and_fill_tensor(
                    ov::element::f32,
                    (param.wtype == ov::element::u8) ? ov::Shape{OC, 1} : ov::Shape{OC, IC / IC_group_size, 1},
                    in_data);
                auto scale_per_oc = std::make_shared<ov::op::v0::Constant>(tensor_scale_per_oc);

                auto weight_deq = std::make_shared<ov::op::v1::Multiply>(weight_f32, scale_per_oc);
                if (param.wtype == ov::element::i8)
                    return weight_deq;

                auto weight_shape = std::make_shared<ov::op::v0::Constant>(
                    ov::element::i32,
                    ov::Shape{2},
                    std::vector<int>{static_cast<int>(OC), static_cast<int>(IC)});
                return std::make_shared<ov::opset10::Reshape>(weight_deq, weight_shape, false);
            }
            if (param.wtype == ov::element::f16) {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = -0.5;
                in_data.range = 1;
                in_data.resolution = resolution;
                auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::f16, ov::Shape{OC, IC}, in_data);
                auto weight_const_f16 = std::make_shared<ov::op::v0::Constant>(tensor);
                return std::make_shared<ov::op::v0::Convert>(weight_const_f16, ov::element::f32);
            }
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -0.5;
            in_data.range = 1;
            in_data.resolution = resolution;
            auto tensor = ov::test::utils::create_and_fill_tensor(param.wtype, ov::Shape{OC, IC}, in_data);
            return std::make_shared<ov::op::v0::Constant>(tensor);
        };

        auto src_abs = std::make_shared<ov::opset10::Abs>(src);
        auto sparsity_threshold = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                         ov::Shape{1, 1, 1},
                                                                         std::vector<float>{param.sparsity_threshold});
        auto lessEqual = std::make_shared<ov::opset10::LessEqual>(src_abs, sparsity_threshold);

        auto zero_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1}, std::vector<float>{0.0f});
        auto sparse_input = std::make_shared<ov::opset10::Select>(lessEqual, zero_const, src);

        auto fc_weight = create_const(param.OC, param.IC, param.IC_group_size, 100);
        auto fc_output = std::make_shared<ov::op::v0::MatMul>(sparse_input, fc_weight, false, true);

        function = std::make_shared<ov::Model>(ov::NodeVector{fc_output}, ov::ParameterVector{src});
    }

    void check_results() {
        auto exec_model = compiledModel.get_runtime_model();

        int fused_node_found = 0;
        for (const auto& n : exec_model->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (layer_type == "ActSparseFC")
                fused_node_found++;
        }
        ASSERT_EQ(fused_node_found, 1);
    }
};

TEST_P(ActSparseFCFusionTest, CompareWithRefs) {
    run();
    check_results();
}

namespace {

static ov::test::InputShape ishape_1st{ov::PartialShape{-1, -1, 4096 / 4},
                                       {ov::Shape{1, 8, 4096 / 4}, ov::Shape{5, 37, 4096 / 4}}};
static ov::test::InputShape ishape_2nd{ov::PartialShape{-1, -1, 4096 / 4}, {ov::Shape{1, 1, 4096 / 4}}};

const std::vector<ActSparseFCFusionParams> test_params = {
    //{ishape_1st, 4096 / 4, 11008 / 4, ov::element::f16, 0, 0.0f},
    //{ishape_2nd, 4096 / 4, 11008 / 4, ov::element::f16, 0, 0.3f},
    {ishape_1st, 4096 / 4, 11008 / 4, ov::element::i8, 0, 0.0f},
    {ishape_2nd, 4096 / 4, 11008 / 4, ov::element::i8, 0, 0.3f},
    {ishape_1st, 4096 / 4, 11008 / 4, ov::element::u8, 0, 0.0f},
    {ishape_2nd, 4096 / 4, 11008 / 4, ov::element::u8, 0, 0.3f},
    {ishape_1st, 4096 / 4, 11008 / 4, ov::element::u4, 128, 0.0f},
    {ishape_2nd, 4096 / 4, 11008 / 4, ov::element::u4, 128, 0.3f},
    {ishape_1st, 4096 / 4, 11008 / 4, ov::element::i4, 128, 0.0f},
    {ishape_2nd, 4096 / 4, 11008 / 4, ov::element::i4, 128, 0.3f},
};

INSTANTIATE_TEST_SUITE_P(smoke_ActSparseFCFusion,
                         ActSparseFCFusionTest,
                         ::testing::ValuesIn(test_params),
                         ActSparseFCFusionTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
