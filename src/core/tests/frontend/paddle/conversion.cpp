// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extension.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "paddle_utils.hpp"

using namespace ov::frontend;

using PDPDConversionExtensionTest = FrontEndConversionExtensionTest;

static const std::string translator_name = "NewCustomOp";

class PaddleFrontendWrapper : public ov::frontend::paddle::FrontEnd {
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override {
        ov::frontend::paddle::FrontEnd::add_extension(extension);
        EXPECT_NE(std::find(m_conversion_extensions.begin(), m_conversion_extensions.end(), extension),
                  m_conversion_extensions.end())
            << "ConversionExtension is not registered.";
        EXPECT_NE(m_op_translators.find(translator_name), m_op_translators.end())
            << translator_name << " translator is not registered.";
    }
};

static ConversionExtensionFEParam getTestData() {
    ConversionExtensionFEParam res;
    res.m_frontEndName = PADDLE_FE;
    res.m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    res.m_modelName = "relu/relu.pdmodel";
    res.m_translatorName = translator_name;
    res.m_frontend = std::make_shared<PaddleFrontendWrapper>();
    return res;
}

INSTANTIATE_TEST_SUITE_P(PDPDConversionExtensionTest,
                         FrontEndConversionExtensionTest,
                         ::testing::Values(getTestData()),
                         FrontEndConversionExtensionTest::getTestCaseName);
