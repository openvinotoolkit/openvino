// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/serialize_data_flow.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <memory>
#include <string>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/store.hpp"

namespace ov {
namespace test {
namespace snippets {

class LinearIRSerializeDataFlow : public testing::Test {
protected:
    std::string m_xml_path;
    std::string m_bin_path;

    void SetUp() override {
        const auto file_prefix = ov::test::utils::generateTestFilePrefix();
        m_xml_path = file_prefix + ".xml";
        m_bin_path = file_prefix + ".bin";
    }

    void TearDown() override {
        ov::test::utils::removeIRFiles(m_xml_path, m_bin_path);
    }
};

TEST_F(LinearIRSerializeDataFlow, PreservesAllStoreBranches) {
    const auto param_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 6});
    const auto param_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 6});
    const auto store_0 = std::make_shared<ov::snippets::op::Store>(param_0);
    const auto store_1 = std::make_shared<ov::snippets::op::Store>(param_1);
    store_0->set_friendly_name("store_0");
    store_1->set_friendly_name("store_1");
    const auto result = std::make_shared<ov::snippets::op::Result>(store_0, ov::OutputVector{store_1});
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_0, param_1});

    auto factory = std::make_shared<ov::snippets::IShapeInferSnippetsFactory>();
    ov::snippets::lowered::LinearIR linear_ir(model, factory);

    EXPECT_FALSE(ov::snippets::lowered::pass::SerializeDataFlow(m_xml_path).run(linear_ir));

    std::ifstream xml_file(m_xml_path);
    ASSERT_TRUE(xml_file.is_open());
    const std::string xml(std::istreambuf_iterator<char>{xml_file}, std::istreambuf_iterator<char>{});
    EXPECT_NE(xml.find("store_0"), std::string::npos);
    EXPECT_NE(xml.find("store_1"), std::string::npos);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
