// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov::test::behavior {

// Deterministic xorshift32 PRNG so generated weights are reproducible from a fixed seed.
inline uint32_t espcnXorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Signed, per-element varied weights in [-1, 1] * 1/sqrt(fan_in). Centering around 0 and Xavier-scaling by
// fan-in keeps the conv cascade + final Tanh out of saturation, so outputs vary with the input instead of
// collapsing to a constant 1.0 - which is what lets reference comparisons catch stale or cross-request results.
inline std::vector<float> espcnMakeConvWeights(size_t count, size_t fanIn, uint32_t seed) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(fanIn));
    std::vector<float> weights(count);
    uint32_t state = seed ? seed : 1u;
    for (size_t i = 0; i < count; ++i) {
        const float unit = static_cast<float>(espcnXorshift32(state) % 20001u) / 10000.0f - 1.0f;  // [-1, 1]
        weights[i] = unit * scale;
    }
    return weights;
}

inline std::shared_ptr<ov::Model> createESPCNX2Model(ov::Dimension batchDimension = ov::Dimension(1, 2),
                                                     ov::Dimension heightDimension = ov::Dimension(32, 64),
                                                     ov::Dimension widthDimension = ov::Dimension(32, 64),
                                                     bool nhwcLayout = true) {
    const ov::PartialShape inputShape = nhwcLayout
                                            ? ov::PartialShape{batchDimension, heightDimension, widthDimension, 1}
                                            : ov::PartialShape{batchDimension, 1, heightDimension, widthDimension};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    input->set_friendly_name("IteratorGetNext:0");
    input->get_output_tensor(0).set_names({"IteratorGetNext:0"});

    ov::Output<ov::Node> nchwInput = input;
    if (nhwcLayout) {
        input->set_layout("NHWC");
        auto transposeOrder = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        nchwInput = std::make_shared<ov::op::v1::Transpose>(input, transposeOrder);
    } else {
        input->set_layout("NCHW");
    }

    // Tests fill inputs up to a few hundred; normalize into Tanh's responsive range so the output is not a
    // constant saturated 1.0 but varies with the input (needed to detect stale/cross-request results).
    auto inputScale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {1.0f / 128.0f});
    ov::Output<ov::Node> scaledInput = std::make_shared<ov::op::v1::Multiply>(nchwInput, inputScale);

    const auto makeConvAdd = [](const ov::Output<ov::Node>& data,
                                size_t inputChannels,
                                size_t outputChannels,
                                size_t kernelSize,
                                uint32_t weightSeed,
                                float biasValue) -> ov::Output<ov::Node> {
        const size_t count = outputChannels * inputChannels * kernelSize * kernelSize;
        const size_t fanIn = inputChannels * kernelSize * kernelSize;
        auto weights =
            ov::op::v0::Constant::create(ov::element::f32,
                                         ov::Shape{outputChannels, inputChannels, kernelSize, kernelSize},
                                         espcnMakeConvWeights(count, fanIn, weightSeed));
        auto convolution = std::make_shared<ov::op::v1::Convolution>(data,
                                                                     weights,
                                                                     ov::Strides{1, 1},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::Strides{1, 1},
                                                                     ov::op::PadType::SAME_UPPER);
        auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, outputChannels, 1, 1}, {biasValue});
        return std::make_shared<ov::op::v1::Add>(convolution, bias);
    };

    // Distinct seeds per conv give independent, reproducible weight patterns.
    auto firstConv = makeConvAdd(scaledInput, 1, 64, 5, 0x9E3779B1u, 0.0f);
    auto firstRelu = std::make_shared<ov::op::v0::Relu>(firstConv);
    auto secondConv = makeConvAdd(firstRelu, 64, 32, 3, 0x85EBCA77u, 0.0f);
    auto secondRelu = std::make_shared<ov::op::v0::Relu>(secondConv);
    auto thirdConv = makeConvAdd(secondRelu, 32, 4, 3, 0xC2B2AE3Du, 0.0f);
    auto depthToSpace =
        std::make_shared<ov::op::v0::DepthToSpace>(thirdConv,
                                                   ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                                                   2);
    auto output = std::make_shared<ov::op::v0::Tanh>(depthToSpace);
    output->set_friendly_name("NCHW_output");
    output->get_output_tensor(0).set_names({"NCHW_output:0"});

    auto result = std::make_shared<ov::op::v0::Result>(output);
    result->set_layout("NCHW");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "ESPCN_x2");
}

struct DynamicModelConfig {
    ov::Dimension batch;
    ov::Dimension height;
    ov::Dimension width;
    bool nhwcLayout;
};

inline const std::map<std::string, DynamicModelConfig>& espcnModelConfigs() {
    static const std::map<std::string, DynamicModelConfig> configs = {
        {"ESPCN_x2_DynHW_FHD", {ov::Dimension(1), ov::Dimension(10, 1080), ov::Dimension(10, 960), true}},
        {"ESPCN_x2_DynNHW_FHD", {ov::Dimension(1, 10), ov::Dimension(1, 1080), ov::Dimension(10, 960), true}},
        {"ESPCN_x2_DynHW_HD", {ov::Dimension(1), ov::Dimension(10, 1080), ov::Dimension(10, 960), true}},
        {"ESPCN_x2_DynNHW_HD_NCHW", {ov::Dimension(1, 10), ov::Dimension(10, 1080), ov::Dimension(10, 960), false}},
        {"ESPCN_x2_DynN_HD_NCHW", {ov::Dimension(1, 10), ov::Dimension(1080), ov::Dimension(960), false}},
    };
    return configs;
}

inline const DynamicModelConfig& getModelConfig(const std::string& modelName) {
    const auto& configs = espcnModelConfigs();
    const auto it = configs.find(modelName);
    OPENVINO_ASSERT(it != configs.end(), "Unknown ESPCN_x2 model name: ", modelName);
    return it->second;
}

inline std::shared_ptr<ov::Model> createESPCNX2ModelByName(const std::string& modelName) {
    const auto& config = getModelConfig(modelName);
    return createESPCNX2Model(config.batch, config.height, config.width, config.nhwcLayout);
}

inline bool hasDynamicBatch(const std::string& modelName) {
    return getModelConfig(modelName).batch.is_dynamic();
}

inline bool hasOnlyDynamicBatch(const std::string& modelName) {
    const auto& config = getModelConfig(modelName);
    return config.batch.is_dynamic() && !config.height.is_dynamic() && !config.width.is_dynamic();
}

inline ov::Shape makeInputShape(const std::shared_ptr<ov::Model>& model, size_t batch, bool useLargeShape) {
    const ov::PartialShape& partialShape = model->input().get_partial_shape();

    ov::Shape shape;
    shape.reserve(partialShape.size());
    for (size_t index = 0; index < partialShape.size(); ++index) {
        const ov::Dimension& dimension = partialShape[index];
        if (index == 0) {
            shape.push_back(batch);
        } else if (dimension.is_static()) {
            shape.push_back(static_cast<size_t>(dimension.get_length()));
        } else {
            const auto interval = dimension.get_interval();
            const int64_t value =
                useLargeShape ? interval.get_max_val() : std::max(interval.get_min_val(), interval.get_max_val() / 2);
            shape.push_back(static_cast<size_t>(value));
        }
    }
    return shape;
}

inline ov::Shape makeOutputShape(const std::string& modelName, const ov::Shape& inputShape) {
    const auto& config = getModelConfig(modelName);
    const size_t heightIndex = config.nhwcLayout ? 1 : 2;
    const size_t widthIndex = config.nhwcLayout ? 2 : 3;
    return {inputShape[0], 1, inputShape[heightIndex] * 2, inputShape[widthIndex] * 2};
}

}  // namespace ov::test::behavior
