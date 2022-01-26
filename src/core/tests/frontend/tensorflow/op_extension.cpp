// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_extension.hpp"

#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/tensorflow/extension/op.hpp"
#include "openvino/frontend/tensorflow/frontend.hpp"
#include "so_extension.hpp"
#include "tf_utils.hpp"

using namespace ov::frontend;

using TFOpExtensionTest = FrontEndOpExtensionTest;

static const std::string translator_name = "Relu";

class Relu1 : public Relu {
    OPENVINO_FRAMEWORK_MAP(tensorflow);
};

class Relu2 : public Relu {
    OPENVINO_FRAMEWORK_MAP(tensorflow, "Relu2");
};

class Relu3 : public Relu {
    OPENVINO_FRAMEWORK_MAP(tensorflow, "Relu3", {});
};

class Relu4 : public Relu {
    OPENVINO_FRAMEWORK_MAP(tensorflow, "Relu4", {}, {});
};

static OpExtensionFEParam getTestData() {
    OpExtensionFEParam res;
    res.m_frontEndName = TF_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.pb";
    res.m_translatorName = translator_name;
    res.m_extension = std::make_shared<OpExtension<Relu1>>();
    return res;
}

INSTANTIATE_TEST_SUITE_P(TFOpExtensionTest,
                         FrontEndOpExtensionTest,
                         ::testing::Values(getTestData()),
                         FrontEndOpExtensionTest::getTestCaseName);
