// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/frontend/extension/conversion.hpp>

#ifdef ENABLE_OV_ONNX_FRONTEND
#    include <openvino/frontend/onnx/extension/conversion.hpp>
#endif

#ifdef ENABLE_OV_PADDLE_FRONTEND
#    include <openvino/frontend/paddle/extension/conversion.hpp>
#endif

#ifdef ENABLE_OV_TF_FRONTEND
#    include <openvino/frontend/tensorflow/extension/conversion.hpp>
#endif

#include "test_extension.hpp"

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>(
    {std::make_shared<TestExtension1>(),
     std::make_shared<ov::frontend::ConversionExtension>("NewCustomOp_1", CustomTranslatorCommon_1),
     std::make_shared<ov::frontend::ConversionExtension>("NewCustomOp_2", CustomTranslatorCommon_2),
#ifdef ENABLE_OV_ONNX_FRONTEND
     std::make_shared<ov::frontend::onnx::ConversionExtension>("NewCustomOp_3", CustomTranslatorONNX),
#endif

#ifdef ENABLE_OV_PADDLE_FRONTEND
     std::make_shared<ov::frontend::paddle::ConversionExtension>("NewCustomOp_4", CustomTranslatorPaddle),
#endif

#ifdef ENABLE_OV_TF_FRONTEND
     std::make_shared<ov::frontend::tensorflow::ConversionExtension>("NewCustomOp_5", CustomTranslatorTensorflow)
#endif
    }));
