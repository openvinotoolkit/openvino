// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "load_from.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const auto PDPD = "pdpd";

using PDPDCutTest = FrontEndLoadFromTest;

static LoadFromFEParam getTestData()
{
    LoadFromFEParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath = std::string(TEST_PDPD_MODELS);
    res.m_file = "conv2d";
    res.m_files = {"2in_2out/2in_2out.pdmodel", "2in_2out/2in_2out.pdiparams"};
    res.m_stream = "relu/relu.pdmodel";
    res.m_streams = {"2in_2out/2in_2out.pdmodel", "2in_2out/2in_2out.pdiparams"};
    return res;
}

INSTANTIATE_TEST_SUITE_P(PDPDCutTest,
                         FrontEndLoadFromTest,
                         ::testing::Values(getTestData()),
                         FrontEndLoadFromTest::getTestCaseName);