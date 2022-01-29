// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "ie_core.hpp"

#include <openvino/runtime/runtime.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace ov;

#ifndef IR_SERIALIZATION_MODELS_PATH  // should be already defined by cmake
#    error "IR_SERIALIZATION_MODELS_PATH is not defined"
#endif

class SerializationDeterministicityTest : public ::testing::Test {
protected:
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
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

TEST_F(SerializationDeterministicityTest, SerializeToStream) {
    const std::string model =
        CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.xml");
    const std::string weights =
        CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.bin");

    std::stringstream m_out_xml_buf, m_out_bin_buf;
    InferenceEngine::Blob::Ptr binBlob;

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_buf, m_out_bin_buf);

    std::streambuf* pbuf = m_out_bin_buf.rdbuf();
    unsigned long bufSize = m_out_bin_buf.tellp();

    InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {bufSize}, InferenceEngine::Layout::C);
    binBlob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc);
    binBlob->allocate();
    pbuf->sgetn(binBlob->buffer(), bufSize);

    auto result = ie.ReadNetwork(m_out_xml_buf.str(), binBlob);

    ASSERT_TRUE(expected.layerCount() == result.layerCount());
    ASSERT_TRUE(expected.getInputShapes() == result.getInputShapes());
}

TEST_F(SerializationDeterministicityTest, SerializeToBlob) {
    const std::string model =
        CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.xml");
    const std::string weights =
        CommonTestUtils::getModelFromTestModelZoo(IR_SERIALIZATION_MODELS_PATH "add_abc_initializers.bin");

    std::stringstream m_out_xml_buf;
    InferenceEngine::Blob::Ptr m_out_bin_buf;

    InferenceEngine::Core ie;
    auto expected = ie.ReadNetwork(model, weights);
    expected.serialize(m_out_xml_buf, m_out_bin_buf);
    auto result = ie.ReadNetwork(m_out_xml_buf.str(), m_out_bin_buf);

    ASSERT_TRUE(expected.layerCount() == result.layerCount());
    ASSERT_TRUE(expected.getInputShapes() == result.getInputShapes());
}

class DeterministicityTest : public ::testing::Test {
protected:
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_F(DeterministicityTest, IRInputsOrder) {
    const std::vector<std::string> friend_names = {"A", "B", "C"};

    auto a = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 2});
    auto add = std::make_shared<opset8::Add>(a, b);
    auto c = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 2});
    auto relu = std::make_shared<opset8::Relu>(c);
    auto add2 = std::make_shared<opset8::Add>(add, relu);
    auto res = std::make_shared<opset8::Result>(add2);

    a->set_friendly_name(friend_names[0]);
    b->set_friendly_name(friend_names[1]);
    c->set_friendly_name(friend_names[2]);

    auto model = std::make_shared<Model>(NodeVector{res}, ParameterVector{a, b, c});

    ov::Core core;
    auto serialize = pass::Serialize(m_out_xml_path, m_out_bin_path);
    serialize.run_on_function(model);
    auto serialized_func = core.read_model(m_out_xml_path);

    for (size_t i = 0; i < friend_names.size(); i++) {
        auto param = serialized_func->get_parameters()[i];
        ASSERT_EQ(i, serialized_func->get_parameter_index(param));
        ASSERT_STREQ(friend_names[i].c_str(), param->get_friendly_name().c_str());
    }
}

TEST_F(DeterministicityTest, IROutputsOrder) {
    const std::vector<std::string> friend_names = {"D", "E", "F"};

    auto a = std::make_shared<opset8::Parameter>(element::f32, Shape{4, 4});
    auto axis_1 = opset8::Constant::create(element::i64, Shape{}, {1});
    auto split1 = std::make_shared<opset8::Split>(a, axis_1, 2);
    auto res1 = std::make_shared<opset8::Result>(split1->output(0));
    auto relu = std::make_shared<opset8::Relu>(split1->output(1));
    auto split2 = std::make_shared<opset8::Split>(relu, axis_1, 2);
    auto res2 = std::make_shared<opset8::Result>(split2->output(0));
    auto res3 = std::make_shared<opset8::Result>(split2->output(1));

    res1->set_friendly_name(friend_names[0]);
    res2->set_friendly_name(friend_names[1]);
    res3->set_friendly_name(friend_names[2]);

    auto model = std::make_shared<Model>(NodeVector{res1, res2, res3}, ParameterVector{a});

    ov::Core core;
    auto serialize = pass::Serialize(m_out_xml_path, m_out_bin_path);
    serialize.run_on_function(model);
    auto serialized_func = core.read_model(m_out_xml_path);

    for (size_t i = 0; i < friend_names.size(); i++) {
        auto out = serialized_func->get_results()[i];
        ASSERT_EQ(i, serialized_func->get_result_index(out));
        ASSERT_STREQ(friend_names[i].c_str(), out->get_friendly_name().c_str());
    }
}