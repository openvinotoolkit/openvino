// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdio>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/serialize.hpp"
#include "read_ir.hpp"

// A key or value holding quantization codes is only useful if it survives a round-trip through
// IR: the operand exists so a model can carry it, and a model carries it through a file.
class QuantizedSDPAOperandsSerializationTest : public ov::test::TestsCommon {
public:
    std::string m_out_xml_path;
    std::string m_out_bin_path;

    void SetUp() override {
        const std::string prefix = ov::test::utils::generateTestFilePrefix();
        m_out_xml_path = prefix + ".xml";
        m_out_bin_path = prefix + ".bin";
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_F(QuantizedSDPAOperandsSerializationTest, RoundTripKeepsOperandTypes) {
    using namespace ov;

    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 8, 16, 4});
    const auto key = std::make_shared<op::v0::Parameter>(element::u4, PartialShape{1, 8, 32, 4});
    const auto value = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{1, 8, 32, 4});
    const auto sdpa = std::make_shared<op::v13::ScaledDotProductAttention>(query, key, value, false);
    const auto model = std::make_shared<Model>(OutputVector{sdpa}, ParameterVector{query, key, value});

    pass::Serialize(m_out_xml_path, m_out_bin_path).run_on_model(model);
    const auto read = ov::test::readModel(m_out_xml_path, m_out_bin_path);

    const auto read_sdpa = read->get_results()[0]->input_value(0).get_node_shared_ptr();
    ASSERT_NE(ov::as_type_ptr<op::v13::ScaledDotProductAttention>(read_sdpa), nullptr);
    EXPECT_EQ(read_sdpa->get_input_element_type(0), element::f16);
    EXPECT_EQ(read_sdpa->get_input_element_type(1), element::u4);
    EXPECT_EQ(read_sdpa->get_input_element_type(2), element::i8);
    EXPECT_EQ(read_sdpa->get_output_element_type(0), element::f16);
}
