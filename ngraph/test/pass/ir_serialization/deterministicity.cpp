// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include "gtest/gtest.h"
#include "ie_core.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"
#include "util/ngraph_test_utils.hpp"
#include "util/test_common.hpp"

class SerializationDeterministicityTest : public ov::test::TestsCommon {
protected:
    std::string test_name = GetTestName();
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

#ifdef NGRAPH_ONNX_FRONTEND_ENABLE

TEST_F(SerializationDeterministicityTest, BasicModel) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc.xml"});

    auto f = read(model);
    ov::pass::Serialize transform{m_out_xml_path_1, m_out_bin_path_1};
    transform.run_on_function(f);
    ov::pass::Serialize transform_2{m_out_xml_path_2, m_out_bin_path_2};
    transform_2.run_on_function(f);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

TEST_F(SerializationDeterministicityTest, ModelWithMultipleLayers) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/addmul_abc.xml"});

    auto f = read(model);
    ov::pass::Serialize transform{m_out_xml_path_1, m_out_bin_path_1};
    transform.run_on_function(f);
    ov::pass::Serialize transform_2{m_out_xml_path_2, m_out_bin_path_2};
    transform_2.run_on_function(f);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

#endif

TEST_F(SerializationDeterministicityTest, ModelWithMultipleOutputs) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/split_equal_parts_2d.xml"});
    const std::string weights = ov::util::path_join({SERIALIZED_ZOO, "ir/split_equal_parts_2d.bin"});

    auto f = read(model, weights);
    ov::pass::Serialize transform{m_out_xml_path_1, m_out_bin_path_1};
    transform.run_on_function(f);
    ov::pass::Serialize transform_2{m_out_xml_path_2, m_out_bin_path_2};
    transform_2.run_on_function(f);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

TEST_F(SerializationDeterministicityTest, ModelWithConstants) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc_initializers.xml"});
    const std::string weights = ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc_initializers.bin"});

    auto f = read(model, weights);
    ov::pass::Serialize transform{m_out_xml_path_1, m_out_bin_path_1};
    transform.run_on_function(f);
    ov::pass::Serialize transform_2{m_out_xml_path_2, m_out_bin_path_2};
    transform_2.run_on_function(f);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

TEST_F(SerializationDeterministicityTest, SerializeToStream) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc_initializers.xml"});
    const std::string weights = ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc_initializers.bin"});

    std::stringstream m_out_xml_buf, m_out_bin_buf;
    InferenceEngine::Blob::Ptr binBlob;

    auto f = read(model, weights);
    ov::pass::Serialize transform{m_out_xml_buf, m_out_bin_buf};
    transform.run_on_function(f);

    std::streambuf* pbuf = m_out_bin_buf.rdbuf();
    unsigned long bufSize = m_out_bin_buf.tellp();

    InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {bufSize}, InferenceEngine::Layout::C);
    binBlob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc);
    binBlob->allocate();
    pbuf->sgetn(binBlob->buffer(), bufSize);

    InferenceEngine::Core ie;
    auto result = ie.ReadNetwork(m_out_xml_buf.str(), binBlob);

    ASSERT_TRUE(f->get_graph_size() == result.getFunction()->get_graph_size());
}

TEST_F(SerializationDeterministicityTest, SerializeToBlob) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc_initializers.xml"});
    const std::string weights = ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc_initializers.bin"});

    std::stringstream m_out_xml_buf;
    InferenceEngine::Blob::Ptr m_out_bin_buf;

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_buf, m_out_bin_buf);
    auto result = ie.ReadNetwork(m_out_xml_buf.str(), m_out_bin_buf);

    ASSERT_TRUE(expected.layerCount() == result.layerCount());
    ASSERT_TRUE(expected.getInputShapes() == result.getInputShapes());
}
