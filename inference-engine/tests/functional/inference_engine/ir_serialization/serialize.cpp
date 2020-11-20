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

class SerializationTest : public ::testing::Test {
protected:
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_F(SerializationTest, BasicModel_MO) {
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

TEST_F(SerializationTest, BasicModel_ONNXImporter) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "add_abc.prototxt";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_TRUE(success) << message;
}

TEST_F(SerializationTest, ModelWithMultipleOutputs_MO) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "split_equal_parts_2d.xml";
    const std::string weights =
        IR_SERIALIZATION_MODELS_PATH "split_equal_parts_2d.bin";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    // Compare function does not support models with multiple outputs
    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_FALSE(success) << message;
}

TEST_F(SerializationTest, ModelWithMultipleOutputs_ONNXImporter) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "split_equal_parts_2d.prototxt";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    // Compare function does not support models with multiple outputs
    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_FALSE(success) << message;
}

TEST_F(SerializationTest, ModelWithMultipleLayers_MO) {
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

TEST_F(SerializationTest, ModelWithMultipleLayers_ONNXImporter) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "addmul_abc.prototxt";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_TRUE(success) << message;
}

TEST_F(SerializationTest, ModelWithConstants_MO) {
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

TEST_F(SerializationTest, ModelWithConstants_ONNXImporter) {
    const std::string model =
        IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.prototxt";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_TRUE(success) << message;
}

TEST_F(SerializationTest, ExperimentalDetectronROIFeatureExtractor_MO) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH
        "experimental_detectron_roi_feature_extractor.xml";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_TRUE(success) << message;
}

TEST_F(SerializationTest, ExperimentalDetectronDetectionOutput_MO) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH
        "experimental_detectron_detection_output.xml";

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_TRUE(success) << message;
}
