// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <file_utils.h>
#include <ie_api.h>
#include <ie_iextension.h>
#include "common_test_utils/ngraph_test_utils.hpp"
#include "ie_core.hpp"
#include "ngraph/ngraph.hpp"
#include "transformations/serialize.hpp"
#include <ngraph/opsets/opset6.hpp>

class TensorNameSerializationTest : public CommonTestUtils::TestsCommon {
protected:
    std::string test_name = GetTestName() + "_" + GetTimestamp();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_F(TensorNameSerializationTest, SerializeFunctionWithTensorNames) {
    InferenceEngine::Core ie;

    std::shared_ptr<ngraph::Function> function;
    {
        auto parameter = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 10, 10});
        parameter->set_friendly_name("parameter");
        parameter->get_output_tensor(0).set_names({"input"});
        auto relu_prev = std::make_shared<ngraph::opset6::Relu>(parameter);
        relu_prev->set_friendly_name("relu_prev");
        relu_prev->get_output_tensor(0).set_names({"relu_prev_t", "identity_prev_t"});
        auto relu = std::make_shared<ngraph::opset6::Relu>(relu_prev);
        relu->set_friendly_name("relu");
        relu->get_output_tensor(0).set_names({"relu,t", "identity"});
        const ngraph::ResultVector results{std::make_shared<ngraph::opset6::Result>(relu)};
        results[0]->set_friendly_name("out");
        ngraph::ParameterVector params{parameter};
        function = std::make_shared<ngraph::Function>(results, params, "TensorNames");
    }

    InferenceEngine::CNNNetwork expected(function);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction(), true, true, true, true);

    ASSERT_TRUE(success) << message;
}
