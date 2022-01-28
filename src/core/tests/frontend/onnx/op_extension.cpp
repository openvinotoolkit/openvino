// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_extension.hpp"

#include "onnx_utils.hpp"
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/onnx/extension/op.hpp"
#include "openvino/frontend/onnx/frontend.hpp"
#include "so_extension.hpp"

using namespace ov::frontend;

using TFOpExtensionTest = FrontEndOpExtensionTest;

static const std::string translator_name = "Relu";

class Relu1 : public Relu {
    OPENVINO_FRAMEWORK_MAP(onnx)
};

class Relu2 : public Relu {
    OPENVINO_FRAMEWORK_MAP(onnx, "Relu2")
};

class Relu3 : public Relu {
    OPENVINO_FRAMEWORK_MAP(onnx, "Relu3", {})
};

class Relu4 : public Relu {
    OPENVINO_FRAMEWORK_MAP(onnx, "Relu4", {}, {})
};

static OpExtensionFEParam getTestData() {
    OpExtensionFEParam res;
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "controlflow/loop_2d_add.onnx";
    res.m_translatorName = translator_name;
    res.m_frontend = std::make_shared<ov::frontend::onnx::FrontEnd>();
    res.m_extension = std::make_shared<OpExtension<Relu1>>();
    return res;
}

INSTANTIATE_TEST_SUITE_P(TFOpExtensionTest,
                         FrontEndOpExtensionTest,
                         ::testing::Values(getTestData()),
                         FrontEndOpExtensionTest::getTestCaseName);
