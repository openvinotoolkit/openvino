// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"
#include "read_ir.hpp"
#include "util/test_common.hpp"

class SerializationTransformationTest : public ov::test::TestsCommon {
protected:
    std::string test_name = GetTestName();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";
    std::shared_ptr<ngraph::Function> m_function;

    void SetUp() override {
        const std::string model = ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc.xml"});
        const std::string weights = ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc.bin"});
        m_function = ov::test::readIR(model, weights);
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_F(SerializationTransformationTest, DirectInstantiation) {
    ov::pass::Serialize transform{m_out_xml_path, m_out_bin_path};
    transform.run_on_function(m_function);

    std::ifstream xml(m_out_xml_path);
    std::ifstream bin(m_out_bin_path);
    ASSERT_TRUE(xml.good());
    ASSERT_TRUE(bin.good());
}

TEST_F(SerializationTransformationTest, PassManagerInstantiation) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path);
    manager.run_passes(m_function);

    std::ifstream xml(m_out_xml_path);
    std::ifstream bin(m_out_bin_path);
    ASSERT_TRUE(xml.good());
    ASSERT_TRUE(bin.good());
}
