// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/serialize.hpp"
#include "read_ir.hpp"

class TensorNameSerializationTest : public ov::test::TestsCommon {
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

TEST_F(TensorNameSerializationTest, SerializeFunctionWithTensorNames) {
    std::shared_ptr<ov::Model> model;
    {
        auto parameter = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f32, ov::Shape{1, 3, 10, 10});
        parameter->set_friendly_name("parameter");
        parameter->get_output_tensor(0).set_names({"input"});
        auto relu_prev = std::make_shared<ov::opset8::Relu>(parameter);
        relu_prev->set_friendly_name("relu_prev");
        relu_prev->get_output_tensor(0).set_names({"relu_prev_t", "identity_prev_t"});
        auto relu = std::make_shared<ov::opset8::Relu>(relu_prev);
        relu->set_friendly_name("relu");
        relu->get_output_tensor(0).set_names({"relu,t", "identity"});
        const ov::ResultVector results{std::make_shared<ov::opset8::Result>(relu)};
        results[0]->set_friendly_name("out");
        ov::ParameterVector params{parameter};
        model = std::make_shared<ov::Model>(results, params, "TensorNames");
    }

    ov::pass::Serialize(m_out_xml_path, m_out_bin_path).run_on_model(model);
    auto result = ov::test::readModel(m_out_xml_path, m_out_bin_path);

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(result, model);
    EXPECT_TRUE(res.valid) << res.message;
}
