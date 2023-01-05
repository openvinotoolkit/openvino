// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/frontend/extension/conversion.hpp>
#include <openvino/opsets/opset8.hpp>

#ifdef ENABLE_OV_ONNX_FRONTEND
#    include <openvino/frontend/onnx/extension/conversion.hpp>
#    define ONNX_EXT                                                                                      \
        std::make_shared<ov::frontend::onnx::ConversionExtension>("NewCustomOp_3", CustomTranslatorONNX), \
            std::make_shared<ov::frontend::onnx::ConversionExtension>("Relu", ReluToSwishTranslator),
#else
#    define ONNX_EXT
#endif

#ifdef ENABLE_OV_PADDLE_FRONTEND
#    include <openvino/frontend/paddle/extension/conversion.hpp>
#    define PADDLE_EXT                                                                                        \
        std::make_shared<ov::frontend::paddle::ConversionExtension>("NewCustomOp_4", CustomTranslatorPaddle), \
            std::make_shared<ov::frontend::paddle::ConversionExtension>("relu", ReluToSwishTranslatorPDPD),
#else
#    define PADDLE_EXT
#endif

#ifdef ENABLE_OV_TF_FRONTEND
#    include <openvino/frontend/tensorflow/extension/conversion.hpp>
#    define TF_EXT                                                                                                    \
        std::make_shared<ov::frontend::tensorflow::ConversionExtension>("NewCustomOp_5", CustomTranslatorTensorflow), \
            std::make_shared<ov::frontend::tensorflow::ConversionExtension>("Relu", ReluToSwishTranslator),
#else
#    define TF_EXT
#endif

#ifdef ENABLE_OV_TF_LITE_FRONTEND
#    include <openvino/frontend/tensorflow_lite/extension/conversion.hpp>
#    define TF_LITE_EXT                                                                                   \
        std::make_shared<ov::frontend::tensorflow_lite::ConversionExtension>("NewCustomOp_6",             \
                                                                             CustomTranslatorTensorflow), \
            std::make_shared<ov::frontend::tensorflow_lite::ConversionExtension>("RELU", ReluToSwishTranslator),
#else
#    define TF_LITE_EXT
#endif

ov::OutputVector CustomTranslatorCommon_1(const ov::frontend::NodeContext& node) {
    return ov::OutputVector();
}

std::map<std::string, ov::OutputVector> CustomTranslatorCommon_2(const ov::frontend::NodeContext& node) {
    return std::map<std::string, ov::OutputVector>();
}

ov::OutputVector CustomTranslatorTensorflow(const ov::frontend::NodeContext& node) {
    return ov::OutputVector();
}

ov::OutputVector CustomTranslatorONNX(const ov::frontend::NodeContext& node) {
    return ov::OutputVector();
}

ov::OutputVector ReluToSwishTranslator(const ov::frontend::NodeContext& node) {
    return {std::make_shared<ov::opset8::Swish>(node.get_input(0))};
}

std::map<std::string, ov::OutputVector> ReluToSwishTranslatorPDPD(const ov::frontend::NodeContext& node) {
    return {{"Out", {std::make_shared<ov::opset8::Swish>(node.get_input("X"))}}};
}

std::map<std::string, ov::OutputVector> CustomTranslatorPaddle(const ov::frontend::NodeContext& node) {
    return std::map<std::string, ov::OutputVector>();
}

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>(
    {std::make_shared<ov::frontend::ConversionExtension>("NewCustomOp_1", CustomTranslatorCommon_1),
     std::make_shared<ov::frontend::ConversionExtension>("NewCustomOp_2", CustomTranslatorCommon_2),
     ONNX_EXT PADDLE_EXT TF_EXT TF_LITE_EXT}));
