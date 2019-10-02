// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cnn_network_ngraph_impl.hpp>
#include "tests_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/result.hpp>

using namespace testing;
using namespace InferenceEngine;

class NGraphReshapeTests : public TestsCommon {
protected:
    void TearDown() override {}
    void SetUp() override {}
};

TEST_F(NGraphReshapeTests, ReshapeBatchReLU) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));

    {
        ngraph::PartialShape shape({2, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);

        ngraph->replace_parameter(0, param);
        ngraph->validate_nodes_and_infer_types();
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({2, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({2, 3, 22, 22}));
}

TEST_F(NGraphReshapeTests, ReshapeSpatialReLU) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));

    {
        ngraph::PartialShape shape({1, 3, 25, 25});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);

        ngraph->replace_parameter(0, param);
        ngraph->validate_nodes_and_infer_types();
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
}

TEST_F(NGraphReshapeTests, CNNReshapeSpatialReLU) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("data");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));

    details::CNNNetworkNGraphImpl cnnNetwork(ngraph);
    std::map<std::string, std::vector<size_t>> shapes;
    shapes["data"] = {1, 3, 25, 25};

    ASSERT_EQ(StatusCode::OK, cnnNetwork.reshape(shapes, nullptr));

    ASSERT_NE(nullptr, cnnNetwork.getFunction());
    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
}
