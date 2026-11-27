// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

namespace {
using params = std::tuple<ov::element::Type, ov::element::Type>;

class InferencePrecisionTests : public ::testing::TestWithParam<params> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<params> &obj) {
        const auto& [model_precision, inference_precision] = obj.param;
        std::stringstream s;
        s << "model_precision=" << model_precision << "_inference_precison=" << inference_precision;
        return s.str();
    }
};

TEST_P(InferencePrecisionTests, smoke_canSetInferencePrecisionAndInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto core = ov::test::utils::PluginCache::get().core();

    const auto& [model_precision, inference_precision] = GetParam();
    auto function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice({1, 2, 32, 32}, model_precision);
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(function, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(inference_precision)));
    auto req = compiled_model.create_infer_request();
    OV_ASSERT_NO_THROW(req.infer());
}

static const std::vector<params> test_params = {
    {ov::element::f16, ov::element::f32},
    {ov::element::f16, ov::element::f16},
    {ov::element::f16, ov::element::dynamic},
    {ov::element::f32, ov::element::f32},
    {ov::element::f32, ov::element::f16},
    {ov::element::f32, ov::element::dynamic},
};

INSTANTIATE_TEST_SUITE_P(smoke_GPU_BehaviorTests, InferencePrecisionTests, ::testing::ValuesIn(test_params), InferencePrecisionTests::getTestCaseName);

TEST(InferencePrecisionTests, CantSetInvalidInferencePrecision) {
    ov::Core core;

    OV_ASSERT_NO_THROW(core.get_property(ov::test::utils::DEVICE_GPU, ov::hint::inference_precision));
    ASSERT_ANY_THROW(core.set_property(ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::bf16)));
}

TEST(ExecutionModeTest, SetCompileGetInferPrecisionAndExecMode) {
    ov::Core core;

    core.set_property(ov::test::utils::DEVICE_GPU, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
    auto model = ov::test::utils::make_conv_pool_relu();
    {
        auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f32));
        ASSERT_EQ(ov::hint::ExecutionMode::PERFORMANCE, compiled_model.get_property(ov::hint::execution_mode));
        ASSERT_EQ(ov::element::f32, compiled_model.get_property(ov::hint::inference_precision));
    }

    {
        /* ov::hint::inference_precision has higher priority than ov::hint::execution_mode */
        auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16),
                                                                                     ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
        ASSERT_EQ(ov::hint::ExecutionMode::ACCURACY, compiled_model.get_property(ov::hint::execution_mode));
        ASSERT_EQ(ov::element::f16, compiled_model.get_property(ov::hint::inference_precision));
    }

    {
        /* ov::hint::inference_precision has higher priority than ov::hint::execution_mode */
        auto compiled_model = core.compile_model(model,
                                                 ov::test::utils::DEVICE_GPU,
                                                 ov::hint::inference_precision(ov::element::dynamic),
                                                 ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
        ASSERT_EQ(ov::hint::ExecutionMode::PERFORMANCE, compiled_model.get_property(ov::hint::execution_mode));
        ASSERT_EQ(ov::element::dynamic, compiled_model.get_property(ov::hint::inference_precision));
    }

    {
        auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
        ASSERT_EQ(ov::hint::ExecutionMode::ACCURACY, compiled_model.get_property(ov::hint::execution_mode));
        ASSERT_EQ(ov::element::dynamic, compiled_model.get_property(ov::hint::inference_precision));
    }

    {
        auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
        ASSERT_EQ(ov::hint::ExecutionMode::PERFORMANCE, compiled_model.get_property(ov::hint::execution_mode));
        ASSERT_EQ(ov::element::f16, compiled_model.get_property(ov::hint::inference_precision));
    }
}

// DynamicQuantizeFullyConnected is only registered on oneDNN/IMMAD platforms, so skip elsewhere.
TEST(ExecutionModeTest, AccuracyModeDisablesFullyConnectedDynamicQuantization) {
    ov::Core core;
    const auto caps = core.get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
    if (std::find(caps.cbegin(), caps.cend(), ov::intel_gpu::capability::HW_MATMUL) == caps.cend()) {
        GTEST_SKIP() << "FC dynamic quantization requires a GPU with the HW_MATMUL capability";
    }

    // f16 activations x i8 compressed weights (per-channel scale) -> FullyConnectedCompressed
    constexpr size_t K = 512;
    constexpr size_t N = 256;
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, static_cast<int64_t>(K)});
    auto weights = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{N, K}, std::vector<int8_t>(N * K, 3));
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{N, 1}, std::vector<float>(N, 0.01f));
    auto decompressed = std::make_shared<ov::op::v1::Multiply>(weights_f16, scale);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, decompressed, false, true);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(matmul)}, ov::ParameterVector{input});

    auto count_dynamic_quantize = [](const ov::CompiledModel& compiled_model) {
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

    const auto group_size = ov::hint::dynamic_quantization_group_size(32);

    // Control: PERFORMANCE mode must still apply dynamic quantization, otherwise the check below proves nothing.
    auto perf_model = core.compile_model(model,
                                         ov::test::utils::DEVICE_GPU,
                                         ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE),
                                         group_size);
    ASSERT_GT(count_dynamic_quantize(perf_model), 0u);

    auto acc_model = core.compile_model(model,
                                        ov::test::utils::DEVICE_GPU,
                                        ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY),
                                        group_size);
    ASSERT_EQ(count_dynamic_quantize(acc_model), 0u);
}
} // namespace
