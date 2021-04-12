// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../shared/include/basic_api.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using PDPDBasicTest = FrontEndBasicTest;

static const std::vector<std::string> models {
        std::string("conv2d"),
        std::string("conv2d_s/conv2d.pdmodel"),
        std::string("conv2d_relu/conv2d_relu.pdmodel"),
        std::string("2in_2out/2in_2out.pdmodel"),
};

INSTANTIATE_TEST_CASE_P(PDPDBasicTest, FrontEndBasicTest,
                        ::testing::Combine(
                            ::testing::Values(PDPD),
                            ::testing::Values(PATH_TO_MODELS),
                            ::testing::ValuesIn(models)),
                        FrontEndBasicTest::getTestCaseName);