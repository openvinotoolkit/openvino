// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "basic_api.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";

using PDPDBasicTest = FrontEndBasicTest;

static const std::vector<std::string> models{
    std::string("conv2d"),
    std::string("conv2d_s/conv2d.pdmodel"),
    std::string("conv2d_relu/conv2d_relu.pdmodel"),
    std::string("2in_2out/2in_2out.pdmodel"),
    std::string("multi_tensor_split/multi_tensor_split.pdmodel"),
    std::string("2in_2out_dynbatch/2in_2out_dynbatch.pdmodel"),
};

INSTANTIATE_TEST_SUITE_P(PDPDBasicTest,
                        FrontEndBasicTest,
                        ::testing::Combine(::testing::Values(PDPD),
                                           ::testing::Values(std::string(TEST_PDPD_MODELS)),
                                           ::testing::ValuesIn(models)),
                        FrontEndBasicTest::getTestCaseName);
