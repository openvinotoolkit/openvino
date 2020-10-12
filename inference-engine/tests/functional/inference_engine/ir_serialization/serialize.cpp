// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include <gtest/gtest.h>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "pugixml.hpp"
#include <ie_core.hpp>
#include <ngraph/ngraph.hpp>

#ifndef IR_SERIALIZATION_MODELS_PATH
#define IR_SERIALIZATION_MODELS_PATH ""
#endif

class SerializationTest : public ::testing::Test {
protected:
  std::string m_out_xml_path = "tmp.xml";
  std::string m_out_bin_path = "tmp.bin";

  void TearDown() override {
#if 0 // TODO: remove debug code
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

TEST_F(SerializationTest, DISABLED_ModelWithMultipleOutputs) {
  const std::string model =
      IR_SERIALIZATION_MODELS_PATH "split_equal_parts_2d.xml";
  const std::string weights =
      IR_SERIALIZATION_MODELS_PATH "split_equal_parts_2d.bin";

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

TEST_F(SerializationTest, DISABLED_ModelWithMultipleLayers) {
  FAIL() << "not implemented";
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
