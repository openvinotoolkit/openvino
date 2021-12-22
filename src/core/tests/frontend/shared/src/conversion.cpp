// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/extension/conversion.hpp>
#include <openvino/op/util/framework_node.hpp>

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
class ConversionExtensionMock : public ov::frontend::ConversionExtension<T> {
public:
    ConversionExtensionMock(const std::string& op_type, const ov::frontend::CreatorFunction<T>& converter)
        : ov::frontend::ConversionExtension<T>(op_type, converter) {}

    ~ConversionExtensionMock() override = default;
};

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndConversionExtensionTest, TestConversionExtensionMock) {
    std::shared_ptr<ov::Model> function;
    ov::frontend::FrontEnd::Ptr m_frontEnd;
    ov::frontend::InputModel::Ptr m_inputModel;
    bool invoked = false;
    m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName);
    std::string op_type = "NewCustomOp";
    if (m_param.m_frontEndName == "paddle") {
         m_frontEnd->add_extension(std::make_shared<ConversionExtensionMock<std::map<std::string, ov::OutputVector>>>(
                 op_type,
                        [&](const NodeContext<std::map<std::string, ov::OutputVector>>& node) -> std::map<std::string, ov::OutputVector> {
                            invoked = true;
                            return {};
                        }));
    } else {
        m_frontEnd->add_extension(std::make_shared<ConversionExtensionMock<ov::OutputVector>>(
                op_type,
            [&](const ov::frontend::NodeContext<ov::OutputVector>& node) -> ov::OutputVector {
                invoked = true;
                return {};
            }));
    }
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(m_param.m_modelName));
    ASSERT_NE(m_inputModel, nullptr);
    std::shared_ptr<ov::Model> decoded_model;
    ASSERT_NO_THROW(decoded_model = m_frontEnd->decode(m_inputModel));
    ASSERT_NE(decoded_model, nullptr);
    for (const auto& op : decoded_model->get_ops()) {
        if (auto fw_node = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(op)) {
            auto attrs = fw_node->get_attrs();
            attrs.set_type_name(op_type);
            fw_node->set_attrs(attrs);
            std::cout << "OK" << std::endl;
        }
    }
    m_frontEnd->convert(decoded_model);
}
