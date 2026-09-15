// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sqrt.hpp"

namespace {

class RMSDynamicQuantizeFusion : public testing::WithParamInterface<ov::element::Type>,
                                 virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::element::Type>& info) {
        return info.param.get_type_name();
    }

protected:
    void validate() override {
        const auto fused_outputs = get_plugin_outputs();

        auto unfused_function = create_model(false);
        auto unfused_model = core->compile_model(unfused_function, targetDevice);
        auto unfused_request = unfused_model.create_infer_request();
        const auto& parameters = function->get_parameters();
        for (size_t i = 0; i < parameters.size(); ++i) {
            unfused_request.set_input_tensor(i, inputs.at(parameters[i]));
        }
        unfused_request.infer();

        std::vector<ov::Tensor> unfused_outputs;
        for (size_t i = 0; i < unfused_function->get_output_size(); ++i) {
            unfused_outputs.push_back(unfused_request.get_output_tensor(i));
        }

        const auto count_dynamic_quantize = [](const ov::CompiledModel& compiled_model) {
            size_t count = 0;
            for (const auto& node : compiled_model.get_runtime_model()->get_ordered_ops()) {
                const auto& rt_info = node->get_rt_info();
                const auto layer_type = rt_info.find("layerType");
                if (layer_type != rt_info.end() && layer_type->second.as<std::string>() == "DynamicQuantize") {
                    count++;
                }
            }
            return count;
        };

        ASSERT_EQ(count_dynamic_quantize(compiledModel), 0);
        ASSERT_EQ(count_dynamic_quantize(unfused_model), 1);

        ASSERT_EQ(fused_outputs.size(), 2);
        ASSERT_EQ(unfused_outputs.size(), fused_outputs.size());
        ASSERT_EQ(fused_outputs[0].get_shape(), input_shape);
        ASSERT_EQ(fused_outputs[1].get_shape(), scale_shape);
        for (size_t i = 0; i < fused_outputs.size(); ++i) {
            ov::test::utils::compare(unfused_outputs[i], fused_outputs[i]);
        }
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        input_shape = {1, 1, 32};
        scale_shape = {1, 1, 1};
        function = create_model(true);
    }

private:
    std::shared_ptr<ov::Model> create_model(bool use_reduce_mean) const {
        const auto input_type = ov::element::f16;
        auto input = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto squared = std::make_shared<ov::op::v1::Power>(
            input,
            ov::op::v0::Constant::create(input_type, ov::Shape{}, {2.0f}));

        ov::Output<ov::Node> mean;
        if (use_reduce_mean) { // Matches RMSFusion
            mean = std::make_shared<ov::op::v1::ReduceMean>(squared, axes, true);
        } else {
            auto sum = std::make_shared<ov::op::v1::ReduceSum>(squared, axes, true);
            mean = std::make_shared<ov::op::v1::Multiply>(
                sum,
                ov::op::v0::Constant::create(input_type, ov::Shape{}, {1.0f / 32.0f}));
        }
        auto mean_with_epsilon = std::make_shared<ov::op::v1::Add>(
            mean,
            ov::op::v0::Constant::create(input_type, ov::Shape{}, {1e-6f}));
        auto inverse_rms = std::make_shared<ov::op::v1::Divide>(
            ov::op::v0::Constant::create(input_type, ov::Shape{}, {1.0f}),
            std::make_shared<ov::op::v0::Sqrt>(mean_with_epsilon));
        auto normalized = std::make_shared<ov::op::v1::Multiply>(input, inverse_rms);
        auto gamma = ov::op::v0::Constant::create(input_type, ov::Shape{32}, std::vector<float>(32, 1.0f));
        auto rms = std::make_shared<ov::op::v1::Multiply>(normalized, gamma);

        ov::op::internal::DynamicQuantize::Attributes attributes;
        attributes.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
        attributes.quantization_dt = GetParam();
        attributes.scale_dt = ov::element::f8e8m0;
        attributes.group_sizes = {1, 1, 32};
        attributes.scales_zp_output_order = {0, 1, 2};
        attributes.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        auto dynamic_quantize = std::make_shared<ov::op::internal::DynamicQuantize>(rms, attributes);
        return std::make_shared<ov::Model>(ov::OutputVector{dynamic_quantize->output(0),
                                                             dynamic_quantize->output(1)},
                                            ov::ParameterVector{input});
    }

    ov::Shape input_shape;
    ov::Shape scale_shape;
};

TEST_P(RMSDynamicQuantizeFusion, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_RMSDynamicQuantizeFusion,
                         RMSDynamicQuantizeFusion,
                         ::testing::Values(ov::element::f8e4m3, ov::element::f8e5m2),
                         RMSDynamicQuantizeFusion::getTestCaseName);

}  // namespace
