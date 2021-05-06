// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using assignvalueTestParam = FrontendOpTestParam;
using assignvalueTest = FrontendOpTest;

static assignvalueTestParam assign_value_fp32() {
    assignvalueTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS + "paddle_assign_value/";
    res.m_modelName =    "paddle_assign_value.pdmodel";


    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.0, 1.0, 1.0, 1.0},
                                                       {1.0, 1.0, 1.0, 1.0},
                                                       {1.0, 1.0, 1.0, 1.0},
                                                       {1.0, 1.0, 1.0, 1.0}}}}}.get_vector());


    res.expected_outputs.emplace_back(test::NDArray<float, 4>{{{{{2.0, 2.0, 2.0, 2.0},
                                                       {2.0, 2.0, 2.0, 2.0},
                                                       {2.0, 2.0, 2.0, 2.0},
                                                       {2.0, 2.0, 2.0, 2.0}}}}}.get_vector());

    return res;
}

TEST_P(assignvalueTest, test_assignvalue) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, assignvalueTest,
                        ::testing::Values(
                                assign_value_fp32()
                        ),
                        assignvalueTest::getTestCaseName);
