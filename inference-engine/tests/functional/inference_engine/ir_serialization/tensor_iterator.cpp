// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "ie_core.hpp"
#include "ie_blob.h"
#include "common_test_utils/data_utils.hpp"

#ifndef IR_SERIALIZATION_MODELS_PATH  // should be already defined by cmake
#define IR_SERIALIZATION_MODELS_PATH ""
#endif

class SerializationTensorIteratorTest : public ::testing::Test {
protected:
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_xml_path.c_str());
    }

    void serialize_and_compare(const std::string& model_path, InferenceEngine::Blob::Ptr weights) {
        std::stringstream buffer;
        InferenceEngine::Core ie;

        std::ifstream model(model_path);
        ASSERT_TRUE(model);
        buffer << model.rdbuf();

        auto expected = ie.ReadNetwork(buffer.str(), weights);
        expected.serialize(m_out_xml_path, m_out_bin_path);
        auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

        bool success;
        std::string message;
        std::tie(success, message) = compare_functions(result.getFunction(), expected.getFunction(), true);
        ASSERT_TRUE(success) << message;
    }
};

TEST_F(SerializationTensorIteratorTest, TiResnet) {
    const std::string model_path = IR_SERIALIZATION_MODELS_PATH "ti_resnet.xml";

    size_t weights_size = 8396840;

    auto weights = InferenceEngine::make_shared_blob<uint8_t>(
            InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, {weights_size}, InferenceEngine::Layout::C));
    weights->allocate();
    CommonTestUtils::fill_data(weights->buffer().as<float *>(), weights->size() / sizeof(float));

    auto *data = weights->buffer().as<int64_t *>();
    data[0] = 1;
    data[1] = 512;
    data[1049602] = 1;
    data[1049603] = 1;
    data[1049604] = 512;

    serialize_and_compare(model_path, weights);
}

TEST_F(SerializationTensorIteratorTest, TiNegativeStride) {
    const std::string model_path = IR_SERIALIZATION_MODELS_PATH "ti_negative_stride.xml";

    size_t weights_size = 3149864;

    auto weights = InferenceEngine::make_shared_blob<uint8_t>(
            InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, {weights_size}, InferenceEngine::Layout::C));
    weights->allocate();
    CommonTestUtils::fill_data(weights->buffer().as<float *>(), weights->size() / sizeof(float));

    auto *data = weights->buffer().as<int64_t *>();
    data[0] = 1;
    data[1] = 512;
    data[393730] = 1;
    data[393731] = 1;
    data[393732] = 256;

    serialize_and_compare(model_path, weights);
}
