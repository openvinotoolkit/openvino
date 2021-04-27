// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using transpose2TestParam = FrontendOpTestParam;
using transpose2Test = FrontendOpTest;

static transpose2TestParam transpose2() {
    transpose2TestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "transpose2";  //TODO: compact model/decomposited

    //Inputs inputs;
    // data (1, 1, 4, 4) input tensor
    res.inputs.emplace_back(test::NDArray<float, 4>{{{{0.41149938, 0.8413846, 0.10664962, 0.39433905 },
		                                      {0.25102508, 0.04060893, 0.94616824, 0.51882 },
		                                      {0.12143327, 0.2064155, 0.6016887, 0.6604398 }},
		                                      {{0.03658712, 0.59241855, 0.6489582, 0.7365383 },
		                                      {0.6795765, 0.8855304, 0.6346199, 0.5214592 },
		                                      {0.056553815, 0.6466467, 0.3580794, 0.53899676 }}}
		                                      }
		                                      .get_vector());

    // (1, 1, 2, 2)
    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{0.41149938, 0.8413846, 0.10664962, 0.39433905 },
                                                                 {0.03658712, 0.59241855, 0.6489582, 0.7365383 }},
                                                                 {{0.25102508, 0.04060893, 0.94616824, 0.51882 },
                                                                 {0.6795765, 0.8855304, 0.6346199, 0.5214592 }},
                                                                 {{0.12143327, 0.2064155, 0.6016887, 0.6604398 },
                                                                 {0.056553815, 0.6466467, 0.3580794, 0.53899676 }}}})
                                                                 .get_vector());

    return res;
}

TEST_P(transpose2Test, test_transpose2) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, transpose2Test,
                        ::testing::Values(
                            transpose2()
                        ),                        
                        transpose2Test::getTestCaseName);                                                 
