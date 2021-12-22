// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extension.hpp"
#include "openvino/frontend/tensorflow/frontend.hpp"
#include "tf_utils.hpp"

using namespace ov::frontend;

using TFConversionExtensionTest = FrontEndConversionExtensionTest;

static const std::string translator_name = "NewCustomOp";

class TensorflowFrontendWrapper : public ov::frontend::tensorflow::FrontEnd {
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override {
        ov::frontend::tensorflow::FrontEnd::add_extension(extension);
        EXPECT_NE(std::find(m_conversion_extensions.begin(), m_conversion_extensions.end(), extension),
                  m_conversion_extensions.end())
            << "ConversionExtension is not registered.";
        EXPECT_NE(m_op_translators.find(translator_name), m_op_translators.end())
            << translator_name << " translator is not registered.";
    }
};

static ConversionExtensionFEParam getTestData() {
    ConversionExtensionFEParam res;
    res.m_frontEndName = TF_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.pb";
    res.m_translatorName = translator_name;
    res.m_frontend = std::make_shared<TensorflowFrontendWrapper>();
    return res;
}

INSTANTIATE_TEST_SUITE_P(TFConversionExtensionTest,
                         FrontEndConversionExtensionTest,
                         ::testing::Values(getTestData()),
                         FrontEndConversionExtensionTest::getTestCaseName);
