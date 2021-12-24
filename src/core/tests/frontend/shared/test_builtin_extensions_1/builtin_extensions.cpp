// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/frontend/extension/conversion.hpp>
#include <openvino/frontend/onnx/extension/conversion.hpp>
#include <openvino/frontend/paddle/extension/conversion.hpp>
#include <openvino/frontend/tensorflow/extension/conversion.hpp>

#include "test_extension.hpp"

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>(
    {std::make_shared<TestExtension1>(),
     std::make_shared<ov::frontend::ConversionExtension>("NewCustomOp_1", CustomTranslatorCommon_1),
     std::make_shared<ov::frontend::ConversionExtension>("NewCustomOp_2", CustomTranslatorCommon_2),
     std::make_shared<ov::frontend::onnx::ConversionExtension>("NewCustomOp_3", CustomTranslatorONNX),
     std::make_shared<ov::frontend::paddle::ConversionExtension>("NewCustomOp_4", CustomTranslatorPaddle),
     std::make_shared<ov::frontend::tensorflow::ConversionExtension>("NewCustomOp_5", CustomTranslatorTensorflow)}));
