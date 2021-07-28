// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "ie_core.hpp"

#ifndef IR_SERIALIZATION_MODELS_PATH  // should be already defined by cmake
#define IR_SERIALIZATION_MODELS_PATH ""
#endif

class SerializationDeterministicityTest : public ::testing::Test {
protected:
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
        if (!f1.good()) return false;
        if (!f2.good()) return false;

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

#ifdef NGRAPH_ONNX_IMPORT_ENABLE

TEST_F(SerializationDeterministicityTest, BasicModel) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "add_abc.prototxt";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path_1, m_out_bin_path_1);
    expected.serialize(m_out_xml_path_2, m_out_bin_path_2);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

TEST_F(SerializationDeterministicityTest, ModelWithMultipleLayers) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "addmul_abc.prototxt";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path_1, m_out_bin_path_1);
    expected.serialize(m_out_xml_path_2, m_out_bin_path_2);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

#endif

TEST_F(SerializationDeterministicityTest, ModelWithMultipleOutputs) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "split_equal_parts_2d.xml";
    const std::string weights =
        IR_SERIALIZATION_MODELS_PATH "split_equal_parts_2d.bin";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_path_1, m_out_bin_path_1);
    expected.serialize(m_out_xml_path_2, m_out_bin_path_2);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

TEST_F(SerializationDeterministicityTest, ModelWithConstants) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.xml";
    const std::string weights =
        IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.bin";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_path_1, m_out_bin_path_1);
    expected.serialize(m_out_xml_path_2, m_out_bin_path_2);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

TEST_F(SerializationDeterministicityTest, SerializeToStream) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.xml";
    const std::string weights =
        IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.bin";

    std::stringstream m_out_xml_buf, m_out_bin_buf;
    InferenceEngine::Blob::Ptr binBlob;

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_buf, m_out_bin_buf);

    std::streambuf* pbuf = m_out_bin_buf.rdbuf();
    unsigned long bufSize = m_out_bin_buf.tellp();

    InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8,
                                           { bufSize }, InferenceEngine::Layout::C);
    binBlob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc);
    binBlob->allocate();
    pbuf->sgetn(binBlob->buffer(), bufSize);

    auto result = ie.ReadNetwork(m_out_xml_buf.str(), binBlob);

    ASSERT_TRUE(expected.layerCount() == result.layerCount());
    ASSERT_TRUE(expected.getInputShapes() == result.getInputShapes());
}

TEST_F(SerializationDeterministicityTest, SerializeToBlob) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.xml";
    const std::string weights =
        IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.bin";

    std::stringstream m_out_xml_buf;
    InferenceEngine::Blob::Ptr m_out_bin_buf;

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_buf, m_out_bin_buf);
    auto result = ie.ReadNetwork(m_out_xml_buf.str(), m_out_bin_buf);

    ASSERT_TRUE(expected.layerCount() == result.layerCount());
    ASSERT_TRUE(expected.getInputShapes() == result.getInputShapes());
}
