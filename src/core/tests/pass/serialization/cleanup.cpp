// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/serialize.hpp"
#include "util/test_common.hpp"

class SerializationCleanupTest : public ov::test::TestsCommon {
protected:
    const std::string test_name = GetTestName() + "_" + GetTimestamp();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

namespace {
std::shared_ptr<ngraph::Function> CreateTestFunction(const std::string& name, const ngraph::PartialShape& ps) {
    const auto param = std::make_shared<ov::opset8::Parameter>(ov::element::f16, ps);
    const auto convert = std::make_shared<ov::opset8::Convert>(param, ov::element::f32);
    const auto result = std::make_shared<ov::opset8::Result>(convert);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, name);
}
}  // namespace

TEST_F(SerializationCleanupTest, SerializationShouldWork) {
    const auto f = CreateTestFunction("StaticFunction", ngraph::PartialShape{2, 2});

    ov::pass::Serialize(m_out_xml_path, m_out_bin_path).run_on_model(f);

    // .xml & .bin files should be present
    ASSERT_TRUE(std::ifstream(m_out_xml_path, std::ios::in).good());
    ASSERT_TRUE(std::ifstream(m_out_bin_path, std::ios::in).good());
}

TEST_F(SerializationCleanupTest, SerializationShouldWorkWithDynamicFunction) {
    const auto f = CreateTestFunction("DynamicFunction", ngraph::PartialShape{ngraph::Dimension()});

    ov::pass::Serialize(m_out_xml_path, m_out_bin_path).run_on_model(f);

    // .xml & .bin files should be present
    ASSERT_TRUE(std::ifstream(m_out_xml_path, std::ios::in).good());
    ASSERT_TRUE(std::ifstream(m_out_bin_path, std::ios::in).good());
}
