// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using concatTestParam = FrontendOpTestParam;
using concatTest = FrontendOpTest;

static concatTestParam concat_axis() {
    concatTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "paddle_concat_axis";

    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.0, 1.0, 1.0, 1.0},
                                                              {1.0, 1.0, 1.0, 1.0},
                                                              {1.0, 1.0, 1.0, 1.0},
                                                              {1.0, 1.0, 1.0, 1.0}}}}}.get_vector());

    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{0.0, 0.0, 0.0, 0.0},
                                                              {0.0, 0.0, 0.0, 0.0},
                                                              {0.0, 0.0, 0.0, 0.0},
                                                              {0.0, 0.0, 0.0, 0.0}}}}}.get_vector());



    res.expected_outputs.emplace_back(test::NDArray<float, 4>{{{{1.0, 1.0, 1.0, 1.0},
                                                                       {1.0, 1.0, 1.0, 1.0},
                                                                       {1.0, 1.0, 1.0, 1.0},
                                                                       {1.0, 1.0, 1.0, 1.0},
                                                                       {0.0, 0.0, 0.0, 0.0},
                                                                       {0.0, 0.0, 0.0, 0.0},
                                                                       {0.0, 0.0, 0.0, 0.0},
                                                                       {0.0, 0.0, 0.0, 0.0}}}}.get_vector());
    return res;
}

static concatTestParam paddle_concat_minus_axis() {
    concatTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "concat_minus_axis";

    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.0, 1.0, 1.0, 1.0},
                                                              {1.0, 1.0, 1.0, 1.0},
                                                              {1.0, 1.0, 1.0, 1.0},
                                                              {1.0, 1.0, 1.0, 1.0}}}}}.get_vector());

    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{0.0, 0.0, 0.0, 0.0},
                                                              {0.0, 0.0, 0.0, 0.0},
                                                              {0.0, 0.0, 0.0, 0.0},
                                                              {0.0, 0.0, 0.0, 0.0}}}}}.get_vector());



    res.expected_outputs.emplace_back(test::NDArray<float, 4>{{{{{1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00 },
                                                                       {1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00 },
                                                                       {1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00 },
                                                                       {1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00 }}}}}.get_vector());
    return res;
}

TEST_P(concatTest, test_concat) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, concatTest,
                        ::testing::Values(
                                concat_axis(),
                                concat_minus_axis()
                        ),
                        concatTest::getTestCaseName);
