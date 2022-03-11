// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/frontend/extension/conversion.hpp>

#ifdef ENABLE_OV_ONNX_FRONTEND
#    include <openvino/frontend/onnx/extension/conversion.hpp>
#    define ONNX_EXT                                                                                      \
        std::make_shared<ov::frontend::onnx::ConversionExtension>("NewCustomOp_3", CustomTranslatorONNX), \
            std::make_shared<ov::frontend::onnx::ConversionExtension>("Relu", ReluToSwishTranslatorONNX),
#else
#    define ONNX_EXT
#endif

#ifdef ENABLE_OV_PADDLE_FRONTEND
#    include <openvino/frontend/paddle/extension/conversion.hpp>
#    define PADDLE_EXT \
        std::make_shared<ov::frontend::paddle::ConversionExtension>("NewCustomOp_4", CustomTranslatorPaddle),
#else
#    define PADDLE_EXT
#endif

#ifdef ENABLE_OV_TF_FRONTEND
#    include <openvino/frontend/tensorflow/extension/conversion.hpp>
#    define TF_EXT \
        std::make_shared<ov::frontend::tensorflow::ConversionExtension>("NewCustomOp_5", CustomTranslatorTensorflow)
#else
#    define TF_EXT
#endif

#include "test_extension.hpp"

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>(
    {std::make_shared<TestExtension1>(),
     std::make_shared<ov::frontend::ConversionExtension>("NewCustomOp_1", CustomTranslatorCommon_1),
     std::make_shared<ov::frontend::ConversionExtension>("NewCustomOp_2", CustomTranslatorCommon_2),
     ONNX_EXT PADDLE_EXT TF_EXT}));
