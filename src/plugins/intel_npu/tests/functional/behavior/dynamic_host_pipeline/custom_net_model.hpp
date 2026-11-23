// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/preprocess/pre_post_processor.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov::test::behavior {

inline std::shared_ptr<ov::Model> createCustomNetModel(bool dynamicBatch = false) {
    const ov::Dimension batchDimension = dynamicBatch ? ov::Dimension(1, 10) : ov::Dimension(1);
    const ov::PartialShape inputShape{batchDimension, 16, ov::Dimension(1, 1280), ov::Dimension(10, 1920)};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputShape);
    input->set_friendly_name("Parameter_59");

    auto make_conv_add = [](const ov::Output<ov::Node>& data,
                            const std::string& convName,
                            const std::string& addName,
                            float weightValue,
                            float biasValue) -> ov::Output<ov::Node> {
        const std::vector<float> weightValues(16 * 16, weightValue);
        const std::vector<float> biasValues(16, biasValue);

        auto weights = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{16, 16, 1, 1}, weightValues);
        auto conv = std::make_shared<ov::op::v1::Convolution>(data,
                                                              weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1},
                                                              ov::op::PadType::EXPLICIT);
        conv->set_friendly_name(convName);

        auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 16, 1, 1}, biasValues);
        auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
        add->set_friendly_name(addName);
        return add;
    };

    auto x = make_conv_add(input, "Convolution_61", "Add_63", 0.01f, 0.001f);
    x = make_conv_add(x, "Convolution_65", "Add_67", 0.011f, 0.001f);

    auto relu68 = std::make_shared<ov::op::v0::Relu>(x);
    relu68->set_friendly_name("Relu_68");
    x = relu68;

    x = make_conv_add(x, "Convolution_70", "Add_72", 0.012f, 0.001f);
    auto relu73 = std::make_shared<ov::op::v0::Relu>(x);
    relu73->set_friendly_name("Relu_73");
    x = relu73;

    x = make_conv_add(x, "Convolution_75", "Add_77", 0.013f, 0.001f);
    auto relu78 = std::make_shared<ov::op::v0::Relu>(x);
    relu78->set_friendly_name("Relu_78");
    x = relu78;

    x = make_conv_add(x, "Convolution_82", "Add_84", 0.014f, 0.001f);
    auto relu85 = std::make_shared<ov::op::v0::Relu>(x);
    relu85->set_friendly_name("Relu_85");
    x = relu85;

    x = make_conv_add(x, "Convolution_87", "Add_89", 0.015f, 0.001f);
    auto relu90 = std::make_shared<ov::op::v0::Relu>(x);
    relu90->set_friendly_name("Relu_90");
    x = relu90;

    x = make_conv_add(x, "Convolution_92", "Add_94", 0.016f, 0.001f);
    auto relu95 = std::make_shared<ov::op::v0::Relu>(x);
    relu95->set_friendly_name("Relu_95");
    x = relu95;

    auto multiplyScale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 16, 1, 1}, {0.5f});
    auto multiply97 = std::make_shared<ov::op::v1::Multiply>(x, multiplyScale);
    multiply97->set_friendly_name("Multiply_97");

    auto add98 = std::make_shared<ov::op::v1::Add>(multiply97, multiply97);
    add98->set_friendly_name("Add_98");

    x = make_conv_add(add98, "Convolution_100", "Add_102", 0.017f, 0.001f);

    auto result = std::make_shared<ov::op::v0::Result>(x);
    result->set_friendly_name("Result_104");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "CustomNet");

    // making input and output to be NHWC
    auto preProc = ov::preprocess::PrePostProcessor(model);
    preProc.input(0).tensor().set_layout("NHWC");
    preProc.input(0).model().set_layout("NCHW");
    preProc.output(0).tensor().set_layout("NHWC");
    preProc.output(0).model().set_layout("NCHW");

    model = preProc.build();

    return model;
}

inline const std::map<std::string, bool>& customNetModelConfigs() {
    // Model name -> dynamic batch.
    static const std::map<std::string, bool> configs = {
        {"CustomNet_DynHW", false},
        {"CustomNet_DynNHW", true},
    };
    return configs;
}

inline bool hasDynamicBatch(const std::string& modelName) {
    const auto& configs = customNetModelConfigs();
    const auto it = configs.find(modelName);
    OPENVINO_ASSERT(it != configs.end(), "Unknown CustomNet model name: ", modelName);
    return it->second;
}

// CustomNet always has dynamic H and W.
inline bool hasOnlyDynamicBatch(const std::string& modelName) {
    hasDynamicBatch(modelName);
    return false;
}

inline std::shared_ptr<ov::Model> createCustomNetModelByName(const std::string& modelName) {
    return createCustomNetModel(hasDynamicBatch(modelName));
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

// 1x1 convolutions with 16 output channels and NHWC in/out: output shape equals input shape.
inline ov::Shape makeOutputShape(const std::string& modelName, const ov::Shape& inputShape) {
    hasDynamicBatch(modelName);
    return inputShape;
}

}  // namespace ov::test::behavior
