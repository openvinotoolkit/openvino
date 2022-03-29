// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "openvino/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"
#include "read_ir.hpp"
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

#ifdef ENABLE_OV_ONNX_FRONTEND

TEST_F(SerializationDeterministicityTest, BasicModel) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc.onnx"});

    auto expected = ov::test::readModel(model, "");
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(expected);
    ov::pass::Serialize(m_out_xml_path_2, m_out_bin_path_2).run_on_model(expected);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

TEST_F(SerializationDeterministicityTest, ModelWithMultipleLayers) {
    const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/addmul_abc.onnx"});

    auto expected = ov::test::readModel(model, "");
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(expected);
    ov::pass::Serialize(m_out_xml_path_2, m_out_bin_path_2).run_on_model(expected);

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

    auto expected = ov::test::readModel(model, weights);
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(expected);
    ov::pass::Serialize(m_out_xml_path_2, m_out_bin_path_2).run_on_model(expected);

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

    auto expected = ov::test::readModel(model, weights);
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(expected);
    ov::pass::Serialize(m_out_xml_path_2, m_out_bin_path_2).run_on_model(expected);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}
