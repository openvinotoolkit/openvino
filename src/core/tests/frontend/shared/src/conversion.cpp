// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/extension/conversion.hpp>

#include "conversion_extension.hpp"
#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndConversionExtensionTest::getTestCaseName(
    const testing::TestParamInfo<ConversionExtensionFEParam>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndConversionExtensionTest::SetUp() {
    FrontEndTestUtils::setupTestEnv();
    m_fem = FrontEndManager();  // re-initialize after setting up environment
    initParamTest();
}

void FrontEndConversionExtensionTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = FrontEndTestUtils::make_model_path(m_param.m_modelsPath + m_param.m_modelName);
}

template <class T>
class ConversionExtensionMock : public ov::frontend::ConversionExtensionBase {
public:
    ConversionExtensionMock(const std::string& op_type, const ov::frontend::CreatorFunction<T>& converter)
        : ov::frontend::ConversionExtensionBase(op_type),
          m_converter(converter){};

    ~ConversionExtensionMock() override = default;

private:
    ov::frontend::CreatorFunction<T> m_converter;
};

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndConversionExtensionTest, TestConversionExtensionMock) {
    std::shared_ptr<ov::Model> function;
    {
        ov::frontend::FrontEnd::Ptr m_frontEnd;
        ov::frontend::InputModel::Ptr m_inputModel;
        m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName);
        if (m_param.m_frontEndName != "pdpd") {
            /*            m_frontEnd->add_extension(std::make_shared<ConversionExtensionMock>(
                            "NewOp",
                            [](const NodeContext& node) -> std::map<std::string, ov::OutputVector> {
                                return {};
                            }));*/
        } else {
            m_frontEnd->add_extension(std::make_shared<ConversionExtensionMock<ov::OutputVector>>(
                "NewOp",
                [](const ov::frontend::NodeContext<ov::OutputVector>& node) -> ov::OutputVector {
                    return {};
                }));
        }
    }
}
