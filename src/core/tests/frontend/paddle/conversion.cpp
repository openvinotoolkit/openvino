// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extension.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "openvino/frontend/plugin_loader.hpp"
#include "paddle_utils.hpp"
#include "so_extension.hpp"

using namespace ov::frontend;

using PDPDConversionExtensionTest = FrontEndConversionExtensionTest;

static const std::string translator_name = "relu";

class PaddleFrontendWrapper : public ov::frontend::paddle::FrontEnd {
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override {
        ov::frontend::paddle::FrontEnd::add_extension(extension);

        if (auto conv_ext = std::dynamic_pointer_cast<ConversionExtension>(extension)) {
            EXPECT_NE(std::find(m_conversion_extensions.begin(), m_conversion_extensions.end(), conv_ext),
                      m_conversion_extensions.end())
                << "ConversionExtension is not registered.";
            EXPECT_NE(m_op_translators.find(conv_ext->get_op_type()), m_op_translators.end())
                << conv_ext->get_op_type() << " translator is not registered.";
        } else if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
            EXPECT_EQ(m_telemetry, telemetry) << "TelemetryExtension is not registered.";
        } else if (auto transformation = std::dynamic_pointer_cast<DecoderTransformationExtension>(extension)) {
            EXPECT_NE(std::find(m_transformation_extensions.begin(), m_transformation_extensions.end(), transformation),
                      m_transformation_extensions.end())
                << "DecoderTransformationExtension is not registered.";
        } else if (auto so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
            if (m_shared_object != nullptr) {
                const auto frontend_shared_data = std::static_pointer_cast<FrontEndSharedData>(m_shared_object);
                EXPECT_TRUE(frontend_shared_data) << "Incorrect type of shared object was used";
                const auto extensions = frontend_shared_data->extensions();
                EXPECT_NE(std::find(extensions.begin(), extensions.end(), so_ext), extensions.end())
                    << "SOExtension is not registered.";
            }
        }
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
