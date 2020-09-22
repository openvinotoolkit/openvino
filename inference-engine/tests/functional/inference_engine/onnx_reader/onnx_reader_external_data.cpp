// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <set>
#include <string>
#include <fstream>

#include <ie_blob.h>
#include <ie_core.hpp>
#include <ngraph/ngraph.hpp>

TEST(ONNX_Reader_Tests, ImportModelWithExternalDataFromFile) {
    InferenceEngine::Core ie;
    const std::string binWeights;
    auto cnnNetwork = ie.ReadNetwork(std::string(ONNX_TEST_MODELS) + "onnx_external_data.prototxt", binWeights);
    auto function = cnnNetwork.getFunction();

    int count_additions = 0;
    int count_constants = 0;
    int count_parameters = 0;

    std::shared_ptr<ngraph::Node> external_data_node;
    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_additions += (op_type == "Add" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
        if (op_type == "Constant") {
            count_constants += 1;
            external_data_node = op;
        }
    }

    ASSERT_EQ(function->get_output_size(), 1);
    ASSERT_EQ(std::string(function->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({2, 2}));
    ASSERT_EQ(count_additions, 2);
    ASSERT_EQ(count_constants, 2);
    ASSERT_EQ(count_parameters, 1);

    const auto external_data_node_const = ngraph::as_type_ptr<ngraph::op::Constant>(external_data_node);
    ASSERT_TRUE(external_data_node_const->get_vector<float>() == (std::vector<float>{1, 2, 3, 4}));
}
