// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/common_util.hpp"
#include "read_ir.hpp"

namespace ov::test {
using op::v0::Parameter, op::v0::Result, op::v0::Relu;
using testing::UnorderedElementsAre;

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

TEST_F(TensorNameSerializationTest, model_with_specific_output_names) {
    const auto make_test_model = [] {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 10, 10});
        input->set_friendly_name("input");
        input->output(0).set_names({"input"});
        auto relu = std::make_shared<Relu>(input);
        relu->set_friendly_name("relu");
        relu->output(0).set_names({"relu"});
        auto result = std::make_shared<Result>(relu);
        result->set_friendly_name("output");
        result->output(0).set_names({"output", "identity,output"});
        return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input}, "Specific output names");
    };
    const auto model_comparator = FunctionsComparator::with_default()
                                      .enable(FunctionsComparator::ATTRIBUTES)
                                      .enable(FunctionsComparator::CONST_VALUES);

    const auto ref_model = make_test_model();
    ov::pass::Serialize(m_out_xml_path, m_out_bin_path).run_on_model(ref_model);
    const auto read_model = ov::test::readModel(m_out_xml_path, m_out_bin_path);

    // Check explicitly output names
    EXPECT_THAT(ref_model->output(0).get_node()->get_input_tensor(0).get_names(),
                UnorderedElementsAre("output", "identity,output", "relu"));
    EXPECT_THAT(ref_model->output(0).get_names(), UnorderedElementsAre("output", "identity,output"));
    EXPECT_THAT(read_model->output(0).get_node()->get_input_tensor(0).get_names(),
                UnorderedElementsAre("output", "identity,output", "relu"));
    EXPECT_THAT(read_model->output(0).get_names(), UnorderedElementsAre("output", "identity,output"));

    const auto res = model_comparator.compare(read_model, ref_model);
    EXPECT_TRUE(res.valid) << res.message;
}
}  // namespace ov::test
