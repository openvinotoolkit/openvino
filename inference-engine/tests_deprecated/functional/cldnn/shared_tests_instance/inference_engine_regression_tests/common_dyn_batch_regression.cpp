// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_dyn_batch_regression.hpp"

std::vector<CommonDynBatchFuncTestParams> supportedDynBatchValues = {
    { "GPU", 4, 3 },
    { "GPU", 4, 2 },
    { "GPU", 4, 1 },
    { "GPU", 8, 5 },
    { "GPU", 8, 4 },
    { "GPU", 8, 3 },
};

INSTANTIATE_TEST_CASE_P(FunctionalTest_smoke, TestNoRegressionDynBatchFP32, ValuesIn(supportedDynBatchValues), getTestCaseName);
