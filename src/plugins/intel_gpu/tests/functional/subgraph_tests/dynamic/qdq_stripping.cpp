// Copyright (C) 2018-2026 Intel Corporation
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

using QDQStrippingParams = std::tuple<ov::test::InputShape, ov::element::Type, ov::element::Type>;

class QuantizationParams {
public:
    ov::Output<ov::Node> build_fq(const ov::Output<ov::Node>& input) const {
        auto input_low = ov::op::v0::Constant::create(ov::element::f32, {}, {i_l});
        auto input_high = ov::op::v0::Constant::create(ov::element::f32, {}, {i_h});
        auto output_low = ov::op::v0::Constant::create(ov::element::f32, {}, {o_l});
        auto output_high = ov::op::v0::Constant::create(ov::element::f32, {}, {o_h});
        return std::make_shared<ov::op::v0::FakeQuantize>(input, input_low, input_high, output_low, output_high, 65536);
    }

    ov::Output<ov::Node> build_dq(const ov::Output<ov::Node>& input, const ov::element::Type& quantization_precision) const {
        auto act_zero_point = ov::op::v0::Constant::create(quantization_precision, {}, {zero_point});
        auto act_zp_convert = std::make_shared<ov::op::v0::Convert>(act_zero_point, ov::element::f32);

        auto act_subtract = std::make_shared<ov::op::v1::Subtract>(input, act_zp_convert);
        auto act_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {(i_h - i_l) / (o_h - o_l)});

        return std::make_shared<ov::op::v1::Multiply>(act_subtract, act_scale);
    }

    float i_l;
    float i_h;
    float o_l;
    float o_h;
    int zero_point;
};

class QDQStrippingTest : public testing::WithParamInterface<QDQStrippingParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QDQStrippingParams>& obj) {
        const auto& [input_shape, input_precision, quantization_precision] = obj.param;
        std::ostringstream result;
        result << "input_shape=" << input_shape << "_input_precision=" << input_precision << "_quantization_precision=" << quantization_precision;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(const ov::PartialShape& input_shape, const ov::element::Type& quantization_precision) {
        OPENVINO_ASSERT(quantization_precision == ov::element::i16 || quantization_precision == ov::element::u16,
                        "Only i16 and u16 quantization precisions are supported in the test");
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};
        // Note: these params are taken from the real cases
        static const std::unordered_map<ov::element::Type_t, std::pair<QuantizationParams, QuantizationParams>> quantization_params{
            {ov::element::Type_t::u16, {{0.f, 10.f, 0.f, 65535.f, 0}, {-6.244578838348389f, 6.347373962402344f, 0.f, 65535.f, 32500}}},
            {ov::element::Type_t::i16,
             {{-5.000076293945312f, 4.999923706054688f, -32768.f, 32767.f, 0}, {-6.296072483062744f, 6.295880317687988f, -32768.f, 32767.f, 0}}},
        };

        const auto& q_params = quantization_params.at(quantization_precision);
        const auto& qp_1 = q_params.first;
        auto input_fq = qp_1.build_fq(params[0]);

        auto input_convert1 = std::make_shared<ov::op::v0::Convert>(input_fq, quantization_precision);
        auto input_convert2 = std::make_shared<ov::op::v0::Convert>(input_convert1, ov::element::f32);

        size_t seed = 1;
        auto create_qdq_branch = [&](float weight_scale_value) {
            auto input_dequantized = qp_1.build_dq(input_convert2, quantization_precision);
            ov::test::utils::InputGenerateData weights_gen_data;
            weights_gen_data.seed = seed;
            auto weight_quantized = ov::test::utils::make_constant(ov::element::i8, ov::Shape{32, 3, 3, 3}, weights_gen_data);
            auto weight_convert = std::make_shared<ov::op::v0::Convert>(weight_quantized, ov::element::f32);
            auto weight_scale = ov::test::utils::make_constant(ov::element::f32, {}, std::vector<float>{weight_scale_value});
            auto weight_dequantized = std::make_shared<ov::op::v1::Multiply>(weight_convert, weight_scale);

            auto conv = std::make_shared<ov::op::v1::Convolution>(input_dequantized,
                                                                  weight_dequantized,
                                                                  ov::Strides{1, 1},
                                                                  ov::CoordinateDiff{1, 1},
                                                                  ov::CoordinateDiff{1, 1},
                                                                  ov::Strides{1, 1});

            ov::test::utils::InputGenerateData bias_gen_data(-2.0, 4, 100, seed++);
            auto bias_const = ov::test::utils::make_constant(ov::element::f32, ov::Shape{1, 32, 1, 1}, bias_gen_data);
            auto conv_biased = std::make_shared<ov::op::v1::Add>(conv, bias_const);

            const auto& qp_2 = q_params.second;
            auto fake_quantize = qp_2.build_fq(conv_biased);
            auto act_quantized = std::make_shared<ov::op::v0::Convert>(fake_quantize, quantization_precision);
            auto act_convert = std::make_shared<ov::op::v0::Convert>(act_quantized, ov::element::f32);
            return qp_2.build_dq(act_convert, quantization_precision);
        };

        auto left_branch = create_qdq_branch(1e-3f);
        auto right_branch = create_qdq_branch(1e-4f);
        auto add_branches = std::make_shared<ov::op::v1::Add>(left_branch, right_branch);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{add_branches}, params, "QDQStripping");
        return model;
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [input_shape, input_precision, quantization_precision] = GetParam();
        init_input_shapes({input_shape});
        inType = outType = input_precision;

        // Since the FQ are not executed in a strictly 'fair' manner, and just replaced with clamp ops, a small accuracy deviation is expected.
        abs_threshold = 1e-3f;
        function = init_subgraph(input_shape.first, quantization_precision);
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        auto runtime_model = compiledModel.get_runtime_model();
        ASSERT_TRUE(runtime_model != nullptr) << "Runtime model should not be null";
        size_t quantize_count = 0;
        for (const auto& op : runtime_model->get_ordered_ops()) {
            auto layer_type = op->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (layer_type == std::string("Quantize")) {
                quantize_count++;
            }
        }
        const size_t expected_quantize_count = 0;
        ASSERT_EQ(quantize_count, expected_quantize_count) << "Unexpected Quantize node count.";
    }
};

TEST_P(QDQStrippingTest, Inference) {
    run();
}

const std::vector<ov::test::InputShape> input_shapes = {{{-1, -1, -1, -1}, {{1, 3, 128, 128}}}};
const std::vector<ov::element::Type> input_precisions = {ov::element::f32};
const std::vector<ov::element::Type> quantization_precisions = {ov::element::u16, ov::element::i16};

INSTANTIATE_TEST_SUITE_P(smoke_QDQStripping,
                         QDQStrippingTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(quantization_precisions)),
                         QDQStrippingTest::getTestCaseName);
}  // namespace