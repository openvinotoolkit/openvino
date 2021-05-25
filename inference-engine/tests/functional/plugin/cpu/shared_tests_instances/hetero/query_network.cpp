// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "hetero/query_network.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace {
using namespace HeteroTests;

auto ConvBias = ngraph::builder::subgraph::makeConvBias();

std::pair<std::set<std::string>, std::shared_ptr<ngraph::Function>> unsupportedNode() {
    ngraph::element::Type type = ngraph::element::Type_t::f32;
    auto params = ngraph::builder::makeParams(type, {{1, 3, 24, 24}});
    auto normalize = ngraph::builder::makeNormalizeL2(params[0], std::vector<int64_t>{2, 3}, 1e-4f, ngraph::op::EpsMode::ADD);
    std::shared_ptr<ngraph::Function> fn_ptr = std::make_shared<ngraph::Function>(normalize, params, "UnsupportedNormalizeAxes");
    std::set<std::string> layers;
    return std::make_pair(layers, fn_ptr);
}

INSTANTIATE_TEST_CASE_P(smoke_CPU, QueryNetworkTest,
                        ::testing::Combine(
                                ::testing::Values("CPU", "HETERO:CPU", "MULTI:CPU"),
                                ::testing::Values(QueryNetworkTest::generateParams(ConvBias), unsupportedNode())),
                        QueryNetworkTest::getTestCaseName);
}  // namespace
