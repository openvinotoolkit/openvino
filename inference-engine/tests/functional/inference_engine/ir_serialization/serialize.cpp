// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "ie_core.hpp"

#ifndef IR_SERIALIZATION_MODELS_PATH  // should be already defined by cmake
#define IR_SERIALIZATION_MODELS_PATH ""
#endif

bool binary_equal(const std::string& f1, const std::string& f2) {
    std::ifstream if1{f1, std::ios::binary | std::ios::ate};
    std::ifstream if2{f2, std::ios::binary | std::ios::ate};

    const size_t f1size = if1.tellg();
    const size_t f2size = if2.tellg();
    if (f1size != f2size) {
        return false;
    }

    if1.seekg(0);
    if2.seekg(0);

    return std::equal(std::istream_iterator<char>(if1),
                      std::istream_iterator<char>(),
                      std::istream_iterator<char>(if2));
}
typedef std::tuple<std::string> SerializationParams;

class SerializationTest: public CommonTestUtils::TestsCommon,
                         public testing::WithParamInterface<SerializationParams> {
public:
    std::string m_model_path;
    std::string m_out_xml_path;
    std::string m_out_bin_path;

    void SetUp() override {
        m_model_path = IR_SERIALIZATION_MODELS_PATH + std::get<0>(GetParam());
        // TODO consider std::tmpnam
        std::cout << "TEST START: " << m_model_path << std::endl;
        const std::string test_name =
            "test"; //  ::testing::UnitTest::GetInstance()->current_test_info()->name();
        m_out_xml_path = test_name + ".xml";
        m_out_bin_path = test_name + ".bin";
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_P(SerializationTest, CompareFunctions) {
    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(m_model_path);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(result.getFunction(), expected.getFunction(), true);
    ASSERT_TRUE(success) << message;

//    const auto& model_bin_path = m_model_path.substr(0, m_model_path.length() - 3).append("bin");
//    EXPECT_TRUE(binary_equal(m_out_bin_path, model_bin_path));
}

INSTANTIATE_TEST_CASE_P(IRSerialization, SerializationTest,
        testing::Values(std::make_tuple("add_abc.xml"),
                        std::make_tuple("add_abc_f64.xml"),
                        std::make_tuple("split_equal_parts_2d.xml"),
                        std::make_tuple("addmul_abc.xml"),
                        std::make_tuple("add_abc_initializers.xml"),
                        std::make_tuple("experimental_detectron_roi_feature_extractor.xml"),
                        std::make_tuple("experimental_detectron_detection_output.xml"),
                        std::make_tuple("nms5.xml"),
                        std::make_tuple("shape_of.xml")));

INSTANTIATE_TEST_CASE_P(ONNXSerialization, SerializationTest,
        testing::Values(std::make_tuple("add_abc.prototxt"),
                        std::make_tuple("split_equal_parts_2d.prototxt"),
                        std::make_tuple("addmul_abc.prototxt"),
                        std::make_tuple("add_abc_initializers.prototxt")));
