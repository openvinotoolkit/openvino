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

typedef std::tuple<std::string, std::string> SerializationParams;

class SerializationTest: public CommonTestUtils::TestsCommon,
                         public testing::WithParamInterface<SerializationParams> {
public:
    std::string m_model_path;
    std::string m_binary_path;
    std::string m_out_xml_path;
    std::string m_out_bin_path;

    void SetUp() override {
        m_model_path = IR_SERIALIZATION_MODELS_PATH + std::get<0>(GetParam());
        if (!std::get<1>(GetParam()).empty()) {
            m_binary_path = IR_SERIALIZATION_MODELS_PATH + std::get<1>(GetParam());
        }

        const std::string test_name =  GetTestName() + "_" + GetTimestamp();
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
    InferenceEngine::CNNNetwork expected;

    expected = ie.ReadNetwork(m_model_path, m_binary_path);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(result.getFunction(), expected.getFunction(), true, false, true, true, true);
    ASSERT_TRUE(success) << message;
}

INSTANTIATE_TEST_SUITE_P(IRSerialization, SerializationTest,
        testing::Values(std::make_tuple("add_abc.xml", "add_abc.bin"),
                        std::make_tuple("add_abc_f64.xml", ""),
                        std::make_tuple("add_abc_bin.xml", ""),
                        std::make_tuple("split_equal_parts_2d.xml", "split_equal_parts_2d.bin"),
                        std::make_tuple("addmul_abc.xml", "addmul_abc.bin"),
                        std::make_tuple("add_abc_initializers.xml", "add_abc_initializers.bin"),
                        std::make_tuple("add_abc_initializers.xml", "add_abc_initializers_f32_nan_const.bin"),
                        std::make_tuple("add_abc_initializers_nan_const.xml", "add_abc_initializers_nan_const.bin"),
                        std::make_tuple("add_abc_initializers_u1_const.xml", "add_abc_initializers_u1_const.bin"),
                        std::make_tuple("experimental_detectron_roi_feature_extractor.xml", ""),
                        std::make_tuple("experimental_detectron_roi_feature_extractor_opset6.xml", ""),
                        std::make_tuple("experimental_detectron_detection_output.xml", ""),
                        std::make_tuple("experimental_detectron_detection_output_opset6.xml", ""),
                        std::make_tuple("nms5.xml", "nms5.bin"),
                        std::make_tuple("shape_of.xml", ""),
                        std::make_tuple("pad_with_shape_of.xml", ""),
                        std::make_tuple("conv_with_rt_info.xml", ""),
                        std::make_tuple("loop_2d_add.xml", "loop_2d_add.bin"),
                        std::make_tuple("nms5_dynamism.xml", "nms5_dynamism.bin")));

#ifdef NGRAPH_ONNX_IMPORT_ENABLE

INSTANTIATE_TEST_SUITE_P(ONNXSerialization, SerializationTest,
        testing::Values(std::make_tuple("add_abc.prototxt", ""),
                        std::make_tuple("split_equal_parts_2d.prototxt", ""),
                        std::make_tuple("addmul_abc.prototxt", ""),
                        std::make_tuple("add_abc_initializers.prototxt", "")));

#endif
