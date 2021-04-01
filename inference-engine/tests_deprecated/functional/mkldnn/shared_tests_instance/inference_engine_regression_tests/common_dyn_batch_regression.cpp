// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_dyn_batch_regression.hpp"

std::vector<CommonDynBatchFuncTestParams> supportedDynBatchValues = {
    { "CPU", 4, 3 },
    { "CPU", 4, 2 },
    { "CPU", 4, 1 },
    { "CPU", 8, 5 },
    { "CPU", 8, 4 },
    { "CPU", 8, 3 }
};

INSTANTIATE_TEST_CASE_P(FunctionalTest_smoke, TestNoRegressionDynBatchFP32, ValuesIn(supportedDynBatchValues), getTestCaseName);
