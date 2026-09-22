// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov::test::behavior {

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

    const auto makeConvAdd = [](const ov::Output<ov::Node>& data,
                                size_t inputChannels,
                                size_t outputChannels,
                                size_t kernelSize,
                                float weightValue,
                                float biasValue) -> ov::Output<ov::Node> {
        auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                    ov::Shape{outputChannels, inputChannels, kernelSize, kernelSize},
                                                    {weightValue});
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

    auto firstConv = makeConvAdd(nchwInput, 1, 64, 5, 0.01f, 0.001f);
    auto firstRelu = std::make_shared<ov::op::v0::Relu>(firstConv);
    auto secondConv = makeConvAdd(firstRelu, 64, 32, 3, 0.011f, 0.001f);
    auto secondRelu = std::make_shared<ov::op::v0::Relu>(secondConv);
    auto thirdConv = makeConvAdd(secondRelu, 32, 4, 3, 0.012f, 0.001f);
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
        {"ESPCN_x2_DynHW_FHD2", {ov::Dimension(1), ov::Dimension(10, 2160), ov::Dimension(10, 3840), true}},
        {"ESPCN_x2_DynHW_FHD", {ov::Dimension(1), ov::Dimension(10, 1080), ov::Dimension(10, 1920), true}},
        {"ESPCN_x2_DynNHW_FHD", {ov::Dimension(1, 10), ov::Dimension(1, 1080), ov::Dimension(10, 1920), true}},
        {"ESPCN_x2_DynHW_HD", {ov::Dimension(1), ov::Dimension(10, 1080), ov::Dimension(10, 1280), true}},
        {"ESPCN_x2_DynNHW_HD_NCHW", {ov::Dimension(1, 10), ov::Dimension(10, 720), ov::Dimension(10, 1280), false}},
        {"ESPCN_x2_DynN_HD_NCHW", {ov::Dimension(1, 10), ov::Dimension(720), ov::Dimension(1280), false}},
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