// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ngraph/ngraph.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "pugixml.hpp"

#ifndef IR_SERIALIZATION_MODELS_PATH
#define IR_SERIALIZATION_MODELS_PATH ""
#endif

class SerializationTest : public ::testing::Test {
protected:
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
#if 0  // TODO: remove debug code
    std::remove(m_out_xml_path.c_str());
    std::remove(m_out_bin_path.c_str());
#endif
    }
};

TEST_F(SerializationTest, BasicModel) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "add_abc.xml";
    const std::string weights = IR_SERIALIZATION_MODELS_PATH "add_abc.bin";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_TRUE(success) << message;
}

TEST_F(SerializationTest, ModelWithMultipleOutputs) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "split_equal_parts_2d.xml";
    const std::string weights =
        IR_SERIALIZATION_MODELS_PATH "split_equal_parts_2d.bin";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

// Compare function does not support models with multiple outputs
#ifdef NDEBUG
    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_TRUE(success) << message;
#else
    ASSERT_DEBUG_DEATH(
        compare_functions(result.getFunction(), expected.getFunction()), "");
#endif
}

TEST_F(SerializationTest, ModelWithMultipleLayers) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "addmul_abc.xml";
    const std::string weights = IR_SERIALIZATION_MODELS_PATH "addmul_abc.bin";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_TRUE(success) << message;
}

TEST_F(SerializationTest, ModelWithConstants) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.xml";
    const std::string weights =
        IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.bin";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_TRUE(success) << message;
}
