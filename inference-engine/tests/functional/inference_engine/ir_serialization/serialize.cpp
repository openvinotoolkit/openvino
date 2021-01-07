// Copyright (C) 2020 Intel Corporation
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

constexpr std::size_t maxFileNameLength = 140;
typedef std::tuple<std::string, size_t, std::function<void (InferenceEngine::Blob::Ptr &)>> SerializationParams;

class SerializationTest: public CommonTestUtils::TestsCommon,
                         public testing::WithParamInterface<SerializationParams> {
public:
    std::string m_model_path;
    std::string m_out_xml_path;
    std::string m_out_bin_path;
    size_t weights_size;
    InferenceEngine::Blob::Ptr weights;

    void SetUp() override {
        m_model_path = IR_SERIALIZATION_MODELS_PATH + std::get<0>(GetParam());

        const std::string test_name =  GetTestName().substr(0, maxFileNameLength) + "_" + GetTimestamp();
        m_out_xml_path = test_name + ".xml";
        m_out_bin_path = test_name + ".bin";

        weights_size = std::get<1>(GetParam());

        if (weights_size) {
            std::function<void(InferenceEngine::Blob::Ptr&)> fillBlob = std::get<2>(GetParam());
            weights = InferenceEngine::make_shared_blob<uint8_t>(
                InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, {weights_size}, InferenceEngine::Layout::C));
            weights->allocate();
            CommonTestUtils::fill_data(weights->buffer().as<float *>(), weights->size() / sizeof(float));
            if (fillBlob)
                fillBlob(weights);
        }
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_P(SerializationTest, CompareFunctions) {
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork expected;

    if (weights_size) {
        std::ifstream model(m_model_path);
        std::stringstream buffer;
        buffer << model.rdbuf();
        expected = ie.ReadNetwork(buffer.str(), weights);
    } else {
        expected = ie.ReadNetwork(m_model_path);
    }
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(result.getFunction(), expected.getFunction(), true);
    ASSERT_TRUE(success) << message;
}

INSTANTIATE_TEST_CASE_P(IRSerialization, SerializationTest,
        testing::Values(std::make_tuple("add_abc.xml", 0, nullptr),
                        std::make_tuple("add_abc_f64.xml", 0, nullptr),
                        std::make_tuple("split_equal_parts_2d.xml", 0, nullptr),
                        std::make_tuple("addmul_abc.xml", 0, nullptr),
                        std::make_tuple("add_abc_initializers.xml", 0, nullptr),
                        std::make_tuple("experimental_detectron_roi_feature_extractor.xml", 0, nullptr),
                        std::make_tuple("experimental_detectron_detection_output.xml", 0, nullptr),
                        std::make_tuple("nms5.xml", 0, nullptr),
                        std::make_tuple("ti_negative_stride.xml", 3149864, +[](InferenceEngine::Blob::Ptr& weights) {
                                                        auto *data = weights->buffer().as<int64_t *>();
                                                        data[0] = 1;
                                                        data[1] = 512;
                                                        data[393730] = 1;
                                                        data[393731] = 1;
                                                        data[393732] = 256;}),
                        std::make_tuple("ti_resnet.xml", 8396840, +[](InferenceEngine::Blob::Ptr& weights) {
                                                        auto *data = weights->buffer().as<int64_t *>();
                                                        data[0] = 1;
                                                        data[1] = 512;
                                                        data[1049602] = 1;
                                                        data[1049603] = 1;
                                                        data[1049604] = 512;}),
                        std::make_tuple("shape_of.xml", 0, nullptr)));

INSTANTIATE_TEST_CASE_P(ONNXSerialization, SerializationTest,
        testing::Values(std::make_tuple("add_abc.prototxt", 0, nullptr),
                        std::make_tuple("split_equal_parts_2d.prototxt", 0, nullptr),
                        std::make_tuple("addmul_abc.prototxt", 0, nullptr),
                        std::make_tuple("add_abc_initializers.prototxt", 0, nullptr)));
