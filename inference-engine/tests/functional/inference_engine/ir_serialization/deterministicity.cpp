// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "ie_core.hpp"

#ifndef IR_SERIALIZATION_MODELS_PATH  // should be already defined by cmake
#    error "IR_SERIALIZATION_MODELS_PATH is not defined"
#endif

class SerializationDeterministicityTest : public ::testing::Test {
protected:
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string m_out_xml_path_1 = test_name + "1" + ".xml";
    std::string m_out_bin_path_1 = test_name + "1" + ".bin";
    std::string m_out_xml_path_2 = test_name + "2" + ".xml";
    std::string m_out_bin_path_2 = test_name + "2" + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path_1.c_str());
        std::remove(m_out_xml_path_2.c_str());
        std::remove(m_out_bin_path_1.c_str());
        std::remove(m_out_bin_path_2.c_str());
    }

    bool files_equal(std::ifstream& f1, std::ifstream& f2) {
        if (!f1.good())
            return false;
        if (!f2.good())
            return false;

        while (!f1.eof() && !f2.eof()) {
            if (f1.get() != f2.get()) {
                return false;
            }
        }

        if (f1.eof() != f2.eof()) {
            return false;
        }

        return true;
    }
};

TEST_F(SerializationDeterministicityTest, SerializeToStream) {
    const std::string model =
        CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.xml");
    const std::string weights =
        CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.bin");

    std::stringstream m_out_xml_buf, m_out_bin_buf;
    InferenceEngine::Blob::Ptr binBlob;

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_buf, m_out_bin_buf);

    std::streambuf* pbuf = m_out_bin_buf.rdbuf();
    unsigned long bufSize = m_out_bin_buf.tellp();

    InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {bufSize}, InferenceEngine::Layout::C);
    binBlob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc);
    binBlob->allocate();
    pbuf->sgetn(binBlob->buffer(), bufSize);

    auto result = ie.ReadNetwork(m_out_xml_buf.str(), binBlob);

    ASSERT_TRUE(expected.layerCount() == result.layerCount());
    ASSERT_TRUE(expected.getInputShapes() == result.getInputShapes());
}

TEST_F(SerializationDeterministicityTest, SerializeToBlob) {
    const std::string model =
        CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.xml");
    const std::string weights =
        CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.bin");

    std::stringstream m_out_xml_buf;
    InferenceEngine::Blob::Ptr m_out_bin_buf;

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_buf, m_out_bin_buf);
    auto result = ie.ReadNetwork(m_out_xml_buf.str(), m_out_bin_buf);

    ASSERT_TRUE(expected.layerCount() == result.layerCount());
    ASSERT_TRUE(expected.getInputShapes() == result.getInputShapes());
}
