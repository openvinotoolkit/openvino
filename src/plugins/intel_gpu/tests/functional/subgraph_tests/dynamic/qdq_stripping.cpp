// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using namespace ov::test;
using ov::test::InputShape;

using QDQStrippingParams = std::tuple<ov::test::InputShape, ov::element::Type>;

class QDQStrippingTest : public testing::WithParamInterface<QDQStrippingParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QDQStrippingParams>& obj) {
        const auto& [input_shape, input_precision] = obj.param;
        std::ostringstream result;
        result << "input_shape=" << input_shape << "_input_precision=" << input_precision;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(const ov::PartialShape& input_shape) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

        const float i_l = 0.f, i_h = 10.f, o_l = 0.f, o_h = 65535.f;
        auto input_low = ov::op::v0::Constant::create(ov::element::f32, {}, {i_l});
        auto input_high = ov::op::v0::Constant::create(ov::element::f32, {}, {i_h});
        auto output_low = ov::op::v0::Constant::create(ov::element::f32, {}, {o_l});
        auto output_high = ov::op::v0::Constant::create(ov::element::f32, {}, {o_h});

        auto input_fq = std::make_shared<ov::op::v0::FakeQuantize>(params[0], input_low, input_high, output_low, output_high, 65536);

        auto input_convert1 = std::make_shared<ov::op::v0::Convert>(input_fq, ov::element::u16);
        auto input_convert2 = std::make_shared<ov::op::v0::Convert>(input_convert1, ov::element::f32);

        size_t seed = 1;
        auto create_qdq_branch = [&](float weight_scale_value) {
            auto input_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {(i_h - i_l) / (o_h - o_l)});
            auto input_dequantized = std::make_shared<ov::op::v1::Multiply>(input_convert2, input_scale);

            ov::test::utils::InputGenerateData gen_data;
            gen_data.seed = seed++;
            auto weight_quantized = ov::test::utils::make_constant(ov::element::u8, ov::Shape{32, 3, 3, 3}, gen_data);
            auto weight_convert = std::make_shared<ov::op::v0::Convert>(weight_quantized, ov::element::f32);
            auto weight_scale = ov::test::utils::make_constant(ov::element::f32, {}, gen_data);
            auto weight_dequantized = std::make_shared<ov::op::v1::Multiply>(weight_convert, weight_scale);

            auto conv = std::make_shared<ov::op::v1::Convolution>(input_dequantized,
                                                                  weight_dequantized,
                                                                  ov::Strides{1, 1},
                                                                  ov::CoordinateDiff{1, 1},
                                                                  ov::CoordinateDiff{1, 1},
                                                                  ov::Strides{1, 1});

            auto bias_const = ov::test::utils::make_constant(ov::element::f32, ov::Shape{1, 32, 1, 1}, gen_data);
            auto conv_biased = std::make_shared<ov::op::v1::Add>(conv, bias_const);

            const float conv_i_l = -6.244578838348389f, conv_i_h = 6.347373962402344f, conv_o_l = 0.f, conv_o_h = 65535.f;
            auto conv_input_low = ov::op::v0::Constant::create(ov::element::f32, {}, {conv_i_l});
            auto conv_input_high = ov::op::v0::Constant::create(ov::element::f32, {}, {conv_i_h});
            auto conv_output_low = ov::op::v0::Constant::create(ov::element::f32, {}, {conv_o_l});
            auto conv_output_high = ov::op::v0::Constant::create(ov::element::f32, {}, {conv_o_h});
            auto fake_quantize =
                std::make_shared<ov::op::v0::FakeQuantize>(conv_biased, conv_input_low, conv_input_high, conv_output_low, conv_output_high, 65536);

            auto act_quantized = std::make_shared<ov::op::v0::Convert>(fake_quantize, ov::element::u16);
            auto act_convert = std::make_shared<ov::op::v0::Convert>(act_quantized, ov::element::f32);

            auto act_zero_point = ov::op::v0::Constant::create(ov::element::u16, {}, {32500});
            auto act_zp_convert = std::make_shared<ov::op::v0::Convert>(act_zero_point, ov::element::f32);

            auto act_subtract = std::make_shared<ov::op::v1::Subtract>(act_convert, act_zp_convert);
            auto act_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {(conv_i_h - conv_i_l) / (conv_o_h - conv_o_l)});

            return std::make_shared<ov::op::v1::Multiply>(act_subtract, act_scale);
        };

        auto left_branch = create_qdq_branch(0.01f);
        auto right_branch = create_qdq_branch(0.001f);
        auto add_branches = std::make_shared<ov::op::v1::Add>(left_branch, right_branch);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{add_branches}, params, "QDQStripping");
        return model;
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [input_shape, input_precision] = GetParam();
        init_input_shapes({input_shape});
        inType = outType = input_precision;

        if (input_precision == ov::element::f16) {
            abs_threshold = 1.0f;
        } else {
            abs_threshold = 1e-4f;
        }
        function = init_subgraph(input_shape.first);
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        auto runtime_model = compiledModel.get_runtime_model();
        ASSERT_TRUE(runtime_model != nullptr) << "Runtime model should not be null";
        for (const auto& op : runtime_model->get_ordered_ops()) {
            auto layer_type = op->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            ASSERT_NE(layer_type, "Quantize") << "FakeQuantize node is not expected in the runtime model after QDQ stripping.";
        }
    }
};

TEST_P(QDQStrippingTest, Inference) {
    run();
}

const std::vector<ov::test::InputShape> input_shapes = {{{-1, -1, -1, -1}, {{1, 3, 128, 128}}}};
const std::vector<ov::element::Type> input_precisions = {ov::element::f32};
INSTANTIATE_TEST_SUITE_P(smoke_QDQStripping,
                         QDQStrippingTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes), ::testing::ValuesIn(input_precisions)),
                         QDQStrippingTest::getTestCaseName);
}  // namespace