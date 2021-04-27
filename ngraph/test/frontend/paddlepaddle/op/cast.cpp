// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

/* cast*/
namespace cast {
using castTestParam = FrontendOpTestParam;
using castTest = FrontendOpTest;

static castTestParam cast_test1() {
    castTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "cast_test1";

    // data
    res.inputs.emplace_back(test::NDArray<float, 2>({{1.10, 2.10, 1.00 },
            {3.20, 4.70, 5.60 }}).get_vector());
    res.expected_outputs.emplace_back(test::NDArray<float, 2>({{1.10, 2.10, 1.00 },
            {3.20, 4.70, 5.60 }}).get_vector());

    return res;
}

//static castTestParam cast_test2() {
//    castTestParam res;
//    res.m_frontEndName = PDPD;
//    res.m_modelsPath =   PATH_TO_MODELS;
//    res.m_modelName =    "cast_test2";
//
//    // data
//    res.inputs.emplace_back(test::NDArray<float, 2>({{1.10, 2.10, 1.00 },
//            {3.20, 4.70, 5.60 }}).get_vector());
//    res.expected_outputs.emplace_back(test::NDArray<uint8_t, 2>({{1, 2, 1 },
//            {3, 4, 5 }}).get_vector());
//
//    return res;
//}

TEST_P(castTest, test_cast) {
    validateOp();
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, castTest,
                        ::testing::Values(
                                          cast_test1()
                        ),
                        castTest::getTestCaseName);
}
