// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ov::test {
namespace {

void checkConvolution(size_t inputChannels, size_t outputChannels, bool withBias, bool expectAcl) {
    if (!ov::with_cpu_sve() || !ov::with_cpu_neon_fp16()) {
        GTEST_SKIP() << "This test requires SVE and FP16 support";
    }

    constexpr float inputValue = 0.25f;
    constexpr float weightValue = 0.125f;
    const ov::Shape inputShape{1, inputChannels, 80, 80};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                          ov::Shape{outputChannels, inputChannels, 3, 3},
                                                          weightValue);
    const auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                                weights,
                                                                ov::Strides{2, 2},
                                                                ov::CoordinateDiff{1, 1},
                                                                ov::CoordinateDiff{1, 1},
                                                                ov::Strides{1, 1});
    ov::Output<ov::Node> result = conv;
    if (withBias) {
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, outputChannels, 1, 1}, 0.5f);
        result = std::make_shared<ov::op::v1::Add>(result, bias);
    }

    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f16));
    auto request = compiled.create_infer_request();
    ov::Tensor data(ov::element::f32, inputShape);
    std::fill_n(data.data<float>(), data.get_size(), inputValue);
    request.set_input_tensor(data);
    request.infer();
    const auto output = request.get_output_tensor();
    ASSERT_EQ(output.get_shape(), (ov::Shape{1, outputChannels, 40, 40}));
    const auto* values = output.data<const float>();
    const float biasValue = withBias ? 0.5f : 0.0f;
    const auto channelSum = static_cast<float>(inputChannels) * inputValue * weightValue;
    EXPECT_FLOAT_EQ(values[0], 4 * channelSum + biasValue);
    EXPECT_FLOAT_EQ(values[20 * 40 + 20], 9 * channelSum + biasValue);

    size_t convolutionCount = 0;
    for (const auto& node : compiled.get_runtime_model()->get_ops()) {
        const auto& info = node->get_rt_info();
        if (info.at(ov::exec_model_info::LAYER_TYPE).as<std::string>() != "Convolution") {
            continue;
        }
        ++convolutionCount;
        const auto& implementation = info.at(ov::exec_model_info::IMPL_TYPE).as<std::string>();
        EXPECT_EQ(implementation.find("ref"), std::string::npos) << implementation;
        EXPECT_EQ(info.at(ov::exec_model_info::RUNTIME_PRECISION).as<ov::element::Type>(), ov::element::f16);
        if (expectAcl) {
            EXPECT_NE(implementation.find("_acl"), std::string::npos) << implementation;
        }
    }
    EXPECT_EQ(convolutionCount, 1u);
}

TEST(smoke_ConvSveFP16, SmallConvolutionUsesOptimizedKernel) {
    checkConvolution(16, 32, false, false);
}

TEST(smoke_ConvSveFP16, SmallConvolutionWithBiasUsesOptimizedKernel) {
    checkConvolution(16, 32, true, false);
}

TEST(smoke_ConvSveFP16, OptimizedACLConvolutionKeepsFP16) {
    checkConvolution(64, 64, false, true);
}

}  // namespace
}  // namespace ov::test
