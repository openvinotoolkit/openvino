// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/onnx/extension/conversion.hpp"

#include "common_test_utils/file_utils.hpp"
#include "conversion_extension.hpp"
#include "onnx_utils.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/onnx/frontend.hpp"
#include "openvino/op/add.hpp"

using namespace ov::frontend;

using ONNXConversionExtensionTest = FrontEndConversionExtensionTest;

static const std::string translator_name = "Add";

class ONNXFrontendWrapper : public ov::frontend::onnx::FrontEnd {
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override {
        ov::frontend::onnx::FrontEnd::add_extension(extension);
        if (auto conv_ext = std::dynamic_pointer_cast<ConversionExtension>(extension)) {
            EXPECT_NE(std::find(m_extensions.conversions.begin(), m_extensions.conversions.end(), conv_ext),
                      m_extensions.conversions.end())
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

TEST(ONNXConversionExtensionTest, custom_op_with_custom_domain) {
    const auto ext = std::make_shared<onnx::ConversionExtension>(
        "CustomAdd",
        "custom.op",
        [](const ov::frontend::NodeContext& node) -> ov::OutputVector {
            auto op = std::make_shared<ov::op::v1::Add>(node.get_input(0), node.get_input(1));
            op->get_rt_info().insert({"added_by_extension", true});
            return {op};
        });

    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = onnx::tests::convert_model("missing_op_domain.onnx", ext));

    for (const auto& op : model->get_ops()) {
        if (const auto& add = std::dynamic_pointer_cast<ov::op::v1::Add>(op)) {
            EXPECT_TRUE(add->get_rt_info().count("added_by_extension") == 1);
            return;
        }
    }
    FAIL() << "Expected operation not found in the converted model";
}

TEST(ONNXConversionExtensionTest, custom_op_with_incorrect_numer_of_outputs_exception) {
    const auto ext =
        std::make_shared<onnx::ConversionExtension>("CustomAdd",
                                                    "custom.op",
                                                    [](const ov::frontend::NodeContext& node) -> ov::OutputVector {
                                                        // the default constructor called, the op with 0 output created
                                                        auto op = std::make_shared<ov::op::v1::Add>();
                                                        return {op};
                                                    });

    std::shared_ptr<ov::Model> model;
    ASSERT_THROW(onnx::tests::convert_model("missing_op_domain.onnx", ext), ov::Exception);
}
