// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include "gtest/gtest.h"
#include "ie_core.hpp"
#include "ngraph/ngraph.hpp"
#include "transformations/serialize.hpp"
#include "moc_transformations.hpp"

#ifndef IR_SERIALIZATION_MODELS_PATH  // should be already defined by cmake
#define IR_SERIALIZATION_MODELS_PATH ""
#endif

class SerializationTransformationTest : public ::testing::Test {
protected:
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";
    std::shared_ptr<ngraph::Function> m_function;

    void SetUp() override {
        const std::string model = IR_SERIALIZATION_MODELS_PATH "add_abc.xml";
        const std::string weights = IR_SERIALIZATION_MODELS_PATH "add_abc.bin";
        InferenceEngine::Core ie;
        m_function = ie.ReadNetwork(model, weights).getFunction();
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_F(SerializationTransformationTest, DirectInstantiation) {
    ngraph::pass::Serialize transform{m_out_xml_path, m_out_bin_path};
    transform.run_on_function(m_function);

    std::ifstream xml(m_out_xml_path);
    std::ifstream bin(m_out_bin_path);
    ASSERT_TRUE(xml.good());
    ASSERT_TRUE(bin.good());
}

TEST_F(SerializationTransformationTest, PassManagerInstantiation) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(m_out_xml_path,
                                                   m_out_bin_path);
    manager.run_passes(m_function);

    std::ifstream xml(m_out_xml_path);
    std::ifstream bin(m_out_bin_path);
    ASSERT_TRUE(xml.good());
    ASSERT_TRUE(bin.good());
}

TEST(Serialize, Test) {
    std::string model_path = "/home/gkazanta/openvino/model-optimizer/kso_off.xml";
    InferenceEngine::Core ie;
    auto cnn = ie.ReadNetwork(model_path);
    ngraph::pass::Manager manager;
    manager.register_pass<MOCTransformations>(true);
    manager.register_pass<ngraph::pass::Serialize>("/home/gkazanta/openvino/model-optimizer/kso_off_s.xml",
            "/home/gkazanta/openvino/model-optimizer/kso_off_s.bin");
    manager.run_passes(cnn.getFunction());
}
