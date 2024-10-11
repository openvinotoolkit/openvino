// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/serialize.hpp"

class SerializationCleanupTest : public ov::test::TestsCommon {
protected:
    std::string m_out_xml_path;
    std::string m_out_bin_path;

    void SetUp() override {
        std::string filePrefix = ov::test::utils::generateTestFilePrefix();
        m_out_xml_path = filePrefix + ".xml";
        m_out_bin_path = filePrefix + ".bin";
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

namespace {
std::shared_ptr<ov::Model> create_test_model(const std::string& name, const ov::PartialShape& ps) {
    const auto param = std::make_shared<ov::opset8::Parameter>(ov::element::f16, ps);
    const auto convert = std::make_shared<ov::opset8::Convert>(param, ov::element::f32);
    const auto result = std::make_shared<ov::opset8::Result>(convert);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, name);
}
}  // namespace

TEST_F(SerializationCleanupTest, SerializationShouldWork) {
    const auto f = create_test_model("StaticFunction", ov::PartialShape{2, 2});

    ov::pass::Serialize(m_out_xml_path, m_out_bin_path).run_on_model(f);

    // .xml & .bin files should be present
    ASSERT_TRUE(std::ifstream(m_out_xml_path, std::ios::in).good());
    ASSERT_TRUE(std::ifstream(m_out_bin_path, std::ios::in).good());
}

TEST_F(SerializationCleanupTest, SerializationShouldWorkWithDynamicFunction) {
    const auto f = create_test_model("DynamicFunction", ov::PartialShape{ov::Dimension()});

    ov::pass::Serialize(m_out_xml_path, m_out_bin_path).run_on_model(f);

    // .xml & .bin files should be present
    ASSERT_TRUE(std::ifstream(m_out_xml_path, std::ios::in).good());
    ASSERT_TRUE(std::ifstream(m_out_bin_path, std::ios::in).good());
}

TEST_F(SerializationCleanupTest, SerializationShouldNotWorkWithMissingParameter) {
    const auto model = create_test_model("RemovedParameter", ov::PartialShape{2});
    model->remove_parameter(model->get_parameters()[0]);

    ASSERT_ANY_THROW(ov::pass::Serialize(m_out_xml_path, m_out_bin_path).run_on_model(model));
    EXPECT_FALSE(std::ifstream(m_out_xml_path, std::ios::in).good());
    EXPECT_FALSE(std::ifstream(m_out_bin_path, std::ios::in).good());
}
