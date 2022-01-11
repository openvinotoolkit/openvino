// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extension.hpp"
#include "onnx_utils.hpp"
#include "openvino/frontend/onnx/frontend.hpp"
#include "so_extension.hpp"

using namespace ov::frontend;

using ONNXConversionExtensionTest = FrontEndConversionExtensionTest;

static const std::string translator_name = "Add";

class ONNXFrontendWrapper : public ov::frontend::onnx::FrontEnd {
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override {
        ov::frontend::onnx::FrontEnd::add_extension(extension);
        if (auto conv_ext = std::dynamic_pointer_cast<ConversionExtension>(extension)) {
            EXPECT_NE(std::find(m_conversion_extensions.begin(), m_conversion_extensions.end(), conv_ext),
                      m_conversion_extensions.end())
                << "ConversionExtension is not registered.";
            // TODO: check that operator is actually registered in ONNX FE
            // EXPECT_NE(m_op_translators.find(conv_ext->get_op_type()), m_op_translators.end())
            //                     << conv_ext->get_op_type() << " translator is not registered.";
        } else if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
            EXPECT_EQ(m_extensions.telemetry, telemetry) << "TelemetryExtension is not registered.";
        } else if (auto transformation = std::dynamic_pointer_cast<DecoderTransformationExtension>(extension)) {
            EXPECT_NE(std::find(m_transformation_extensions.begin(), m_transformation_extensions.end(), transformation),
                      m_transformation_extensions.end())
                << "DecoderTransformationExtension is not registered.";
        } else if (auto so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
            EXPECT_NE(std::find(m_other_extensions.begin(), m_other_extensions.end(), so_ext), m_other_extensions.end())
                << "SOExtension is not registered.";
        }
    }
};

static ConversionExtensionFEParam getTestData() {
    ConversionExtensionFEParam res;
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_modelName = "controlflow/loop_2d_add.onnx";
    res.m_translatorName = translator_name;
    res.m_frontend = std::make_shared<ONNXFrontendWrapper>();
    return res;
}

INSTANTIATE_TEST_SUITE_P(ONNXConversionExtensionTest,
                         FrontEndConversionExtensionTest,
                         ::testing::Values(getTestData()),
                         FrontEndConversionExtensionTest::getTestCaseName);
